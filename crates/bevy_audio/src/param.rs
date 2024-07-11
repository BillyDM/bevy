//! Audio parameters

mod smooth;

pub use smooth::SmoothOutput;

use atomic_float::AtomicF32;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc,
};

use self::smooth::Smooth;

/// A value normalized to the range `[0.0, 1.0]`
#[repr(transparent)]
#[derive(Default, Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct NormalVal(f32);

impl NormalVal {
    /// A value of `0.0`
    pub const ZERO: Self = Self(0.0);
    /// A value of `0.5`
    pub const HALF: Self = Self(0.5);
    /// A value of `1.0`
    pub const ONE: Self = Self(1.0);

    /// Construct a new value normalized to the range `[0.0, 1.0]`.
    ///
    /// The value will be clamped to the range `[0.0, 1.0]`.
    pub fn new(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    /// Get the normalized value in the range `[0.0, 1.0]`
    pub fn get(&self) -> f32 {
        self.0
    }
}

impl From<f32> for NormalVal {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl From<NormalVal> for f32 {
    fn from(value: NormalVal) -> Self {
        value.get()
    }
}

/// A good default value to use as `smooth_secs` parameter when creating a [`Param`].
///
/// This specifies that the low-pass parameter smoothing filter should use a period of `5 ms`.
pub const DEFAULT_SMOOTH_SECS: f32 = 5.0 / 1_000.0;

/// The curve used when mapping the normalized value in the range `[0.0, 1.0]` to the
/// desired value.
///
/// For example, it is useful for parameters dealing with decibels to have a mapping
/// curve around `Power(0.15)`. This is so one tick near the top of the slider/knob
/// controlling this parameter causes a small change in dB around `0.0 dB` and one tick
/// on the other end causes a large change in dB around `-100.0 dB`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Curve {
    /// No mapping (use the raw normalized value)
    Normal,
    /// The values are uniformly distributed between `min` and `max`.
    Linear,
    /// The range is skewed by a factor. Values above 1.0 will make the end of the range wider,
    /// while values between 0 and 1 will skew the range towards the start.
    ///
    /// For example, it is useful for parameters dealing with decibels to have a mapping
    /// curve around `0.15`. This is so one tick near the top of the slider/knob
    /// controlling this parameter causes a small change in dB around `0.0 dB` and one tick
    /// on the other end causes a large change in dB around `-100.0 dB`.
    Skewed(f32),
    /// Logarithmic mapping, useful for parameters dealing with frequency in Hz.
    Logarithmic,
}

impl Curve {
    /// Create a new curve from the given skew factor. Positive values make the end of the range
    /// wider while negative make the start of the range wider.
    pub fn from_skew(skew: f32) -> Self {
        Self::Skewed(2.0f32.powf(skew))
    }

    /// Calculate the curve factor that makes a linear gain parameter range appear as if it
    /// was linear when formatted as decibels.
    pub fn from_gain_db(min_db: f32, max_db: f32) -> Self {
        let min_amp = crate::util::db_to_amp_clamped_neg_100_db(min_db);
        let max_amp = crate::util::db_to_amp_clamped_neg_100_db(max_db);
        let middle_db = (max_db + min_db) / 2.0;
        let middle_amp = crate::util::db_to_amp_clamped_neg_100_db(middle_db);

        // Check the Skewed equation in the normalized function below, we need to solve the factor
        // such that the a normalized value of 0.5 resolves to the middle of the range
        let factor = 0.5f32.log((middle_amp - min_amp) / (max_amp - min_amp));

        Self::Skewed(factor)
    }

    /// Map a normalized value to its corresponding unnormalized value.
    pub fn normalize(&self, value: f32, min_value: f32, max_value: f32) -> NormalVal {
        if value <= min_value {
            return NormalVal::ZERO;
        }

        if value >= max_value {
            return NormalVal::ONE;
        }

        let map_linear = |x: f32| -> f32 { (x - min_value) / (max_value - min_value) };

        match self {
            Curve::Normal => NormalVal(value),
            Curve::Linear => NormalVal(map_linear(value)),
            Curve::Skewed(factor) => NormalVal(map_linear(value).powf(*factor)),
            Curve::Logarithmic => {
                let minl = min_value.log2();
                let range = max_value.log2() - minl;
                NormalVal((value.log2() - minl) / range)
            }
        }
    }

    /// Map an unnormalized value to its corresponding normalized value.
    pub fn unnormalize(&self, normalized: NormalVal, min_value: f32, max_value: f32) -> f32 {
        let map_linear = |x: f32| -> f32 {
            let range = max_value - min_value;
            (x * range) + min_value
        };

        match self {
            Curve::Normal => normalized.get(),
            Curve::Linear => map_linear(normalized.get()),
            Curve::Skewed(factor) => map_linear(normalized.get().powf(factor.recip())),
            Curve::Logarithmic => {
                if normalized.get() == 0.0 {
                    return min_value;
                }

                if normalized.get() == 1.0 {
                    return max_value;
                }

                let minl = min_value.log2();
                let range = max_value.log2() - minl;
                2.0f32.powf((normalized.get() * range) + minl)
            }
        }
    }
}

/// The unit of this parameter. This signifies how the value displayed to the end user should
/// differ from the actual value used in DSP.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Unit {
    /// Any kind of unit where the value displayed to the end user is the same value used
    /// in the DSP.
    Generic,
    /// Signifies that the value displayed to the end user should be in decibels and the
    /// value used in the DSP should be in raw amplitude.
    ///
    /// In addition, whenever the dB value is less than or equal to `-100.0 dB`, then the
    /// resulting raw DSP ampilitude value will be clamped to `0.0` (essentially equaling
    /// `-infinity dB`).
    Decibels,
}

impl Unit {
    /// Convert the given unit value to the corresponding raw value used in DSP.
    ///
    /// This is only effective when this unit is not of type `Unit::Generic`.
    pub fn unit_to_dsp(&self, value: f32) -> f32 {
        match self {
            Unit::Decibels => crate::util::db_to_amp_clamped_neg_100_db(value),
            _ => value,
        }
    }

    /// Convert the given raw DSP value to the corresponding unit value.
    ///
    /// This is only effective when this unit is not of type `Unit::Generic`.
    pub fn dsp_to_unit(&self, dsp_value: f32) -> f32 {
        match self {
            Unit::Decibels => crate::util::amp_to_db_clamped_neg_100_db(dsp_value),
            _ => dsp_value,
        }
    }
}

/// A parameter with an `f32` value.
pub struct ParamF32 {
    min_value: f32,
    max_value: f32,
    default_value: f32,

    min_dsp_value: f32,
    max_dsp_value: f32,

    curve: Curve,
    unit: Unit,

    shared_dsp_value: Arc<AtomicF32>,
}

impl ParamF32 {
    /// Create a parameter from its (de-normalized) value.
    ///
    /// * value - The initial (de-normalized) value of the parameter.
    /// * default_value - The default (de-normalized) value of the parameter.
    /// * min_value - The minimum (de-normalized) value of the parameter.
    /// * max_value - The maximum (de-normalized) value of the parameter.
    /// * curve - The [`Curve`] mapping used when converting from the normalized value
    /// in the range `[0.0, 1.0]` to the desired value. If this parameter deals with decibels,
    /// you may use `Param::DEFAULT_SMOOTH_SECS` as a good default.
    ///
    /// # Panics
    /// Panics when `min_value >= max_value`
    pub fn from_value(
        value: f32,
        default_value: f32,
        min_value: f32,
        max_value: f32,
        curve: Curve,
        unit: Unit,
    ) -> Self {
        debug_assert!(min_value < max_value);

        let value = value.clamp(min_value, max_value);
        let dsp_value = unit.unit_to_dsp(value);

        Self {
            min_value,
            max_value,
            default_value,
            min_dsp_value: unit.unit_to_dsp(min_value),
            max_dsp_value: unit.unit_to_dsp(max_value),
            curve,
            unit,
            shared_dsp_value: Arc::new(AtomicF32::new(dsp_value)),
        }
    }

    /// Create a parameter from its normalized value in the range `[0.0, 1.0]`.
    ///
    /// * normalized - The initial normalized value of the parameter in the range `[0.0, 1.0]`.
    /// * default_value - The default (de-normalized) value of the parameter.
    /// * min_value - The minimum (de-normalized) value of the parameter.
    /// * max_value - The maximum (de-normalized) value of the parameter.
    /// * curve - The [`Curve`] mapping used when converting from the normalized value
    /// in the range `[0.0, 1.0]` to the desired value. If this parameter deals with decibels,
    /// you may use `Param::DEFAULT_SMOOTH_SECS` as a good default.
    pub fn from_normalized(
        normalized: NormalVal,
        default_value: f32,
        min_value: f32,
        max_value: f32,
        curve: Curve,
        unit: Unit,
    ) -> Self {
        let dsp_value = curve.unnormalize(normalized, min_value, max_value);

        Self {
            min_value,
            max_value,
            default_value,
            min_dsp_value: unit.unit_to_dsp(min_value),
            max_dsp_value: unit.unit_to_dsp(max_value),
            curve,
            unit,
            shared_dsp_value: Arc::new(AtomicF32::new(dsp_value)),
        }
    }

    /// Activate the parameter and get its realtime processor part.
    pub fn activate(&self) -> ParamF32Proc {
        ParamF32Proc {
            shared_value: Arc::clone(&self.shared_dsp_value),
        }
    }

    /// Activate the parameter with a smoother and gets its realtime processor part.
    pub fn activate_with_smoother(
        &self,
        sample_rate: u32,
        smooth_secs: f32,
        max_block_size: usize,
    ) -> SmoothedParamF32Proc {
        SmoothedParamF32Proc::new(
            Arc::clone(&self.shared_dsp_value),
            sample_rate,
            smooth_secs,
            max_block_size,
        )
    }

    /// Load the normalized value in the range `[0.0, 1.0]`.
    pub fn load_normalized(&self) -> NormalVal {
        let dsp_value = self.shared_dsp_value.load(Ordering::SeqCst);
        self.curve
            .normalize(dsp_value, self.min_dsp_value, self.max_dsp_value)
    }

    /// Load the (un-normalized) value of this parameter.
    pub fn load_value(&self) -> f32 {
        let dsp_value = self.shared_dsp_value.load(Ordering::SeqCst);
        self.unit.dsp_to_unit(dsp_value)
    }

    /// The (un-normalized) default value of the parameter
    pub fn default_value(&self) -> f32 {
        self.default_value
    }

    /// Set the normalized value of this parameter in the range `[0.0, 1.0]`.
    pub fn set_normalized(&mut self, normalized: NormalVal) {
        let dsp_value = self
            .curve
            .unnormalize(normalized, self.min_dsp_value, self.max_dsp_value);
        self.shared_dsp_value.store(dsp_value, Ordering::SeqCst);
    }

    /// Set the (un-normalized) value of this parameter.
    pub fn set_value(&self, value: f32) {
        let value = value.clamp(self.min_value, self.max_value);
        let dsp_value = self.unit.unit_to_dsp(value);
        self.shared_dsp_value.store(dsp_value, Ordering::SeqCst);
    }

    /// The minimum value of this parameter
    pub fn min_value(&self) -> f32 {
        self.min_value
    }

    /// The maximum value of this parameter
    pub fn max_value(&self) -> f32 {
        self.max_value
    }

    /// The [`Curve`] mapping used when converting from the normalized value
    /// in the range `[0.0, 1.0]` to the desired value
    pub fn curve(&self) -> Curve {
        self.curve
    }

    /// The [`Unit`] of this parameter
    pub fn unit(&self) -> Unit {
        self.unit
    }

    /// Convert the given value to the corresponding normalized range `[0.0, 1.0]`
    /// of this parameter.
    pub fn normalize(&self, value: f32) -> NormalVal {
        let dsp_value = self.unit.unit_to_dsp(value);

        self.curve
            .normalize(dsp_value, self.min_dsp_value, self.max_dsp_value)
    }

    /// Convert the given normalized value in the range `[0.0, 1.0]` into the
    /// corresponding value of this parameter.
    pub fn unnormalize(&self, normalized: NormalVal) -> f32 {
        let dsp_value = self
            .curve
            .unnormalize(normalized, self.min_value, self.max_value);
        self.unit.dsp_to_unit(dsp_value)
    }

    /// Get the shared (un-normalized, un-converted) f32 value.
    ///
    /// This can be useful to integrate with various plugin APIs.
    pub fn shared_dsp_value(&self) -> Arc<AtomicF32> {
        Arc::clone(&self.shared_dsp_value)
    }
}

impl Clone for ParamF32 {
    fn clone(&self) -> Self {
        Self {
            min_value: self.min_value,
            max_value: self.max_value,
            default_value: self.default_value,

            min_dsp_value: self.min_dsp_value,
            max_dsp_value: self.max_dsp_value,

            curve: self.curve,
            unit: self.unit,

            shared_dsp_value: Arc::clone(&self.shared_dsp_value),
        }
    }
}

/// The processor for a parameter with an `f32` value (no smoothing)
pub struct ParamF32Proc {
    shared_value: Arc<AtomicF32>,
}

impl ParamF32Proc {
    /// Load the (un-normalized) value.
    ///
    /// Note, this internally contains an atomic load operation, so prefer to only call this
    /// once per process cycle.
    #[inline]
    pub fn load_value(&self) -> f32 {
        self.shared_value.load(Ordering::SeqCst)
    }

    /// Get the shared (un-normalized) f32 value.
    ///
    /// This can be useful to integrate with various plugin APIs.
    pub fn shared_value(&self) -> Arc<AtomicF32> {
        Arc::clone(&self.shared_value)
    }
}

/// The processor for an auto-smoothed parameter with an `f32` value.
pub struct SmoothedParamF32Proc {
    internal: ParamF32Proc,
    smoothed: Smooth,
    prev_value: f32,
}

impl SmoothedParamF32Proc {
    fn new(
        shared_value: Arc<AtomicF32>,
        sample_rate: u32,
        smooth_secs: f32,
        max_block_size: usize,
    ) -> Self {
        let value = shared_value.load(Ordering::SeqCst);

        let mut smoothed = Smooth::new(value, max_block_size);
        smoothed.set_speed(sample_rate, smooth_secs);

        Self {
            internal: ParamF32Proc { shared_value },
            smoothed,
            prev_value: value,
        }
    }

    /// Reset the internal smoothing buffer.
    pub fn reset(&mut self) {
        self.smoothed.reset(self.internal.load_value());
    }

    /// Get the smoothed buffer of values for use in DSP.
    pub fn smoothed(&mut self, frames: usize) -> SmoothOutput {
        let new_val = self.internal.load_value();
        if self.prev_value != new_val {
            self.prev_value = new_val;
            self.smoothed.set(new_val);
        }

        self.smoothed.process(frames);
        self.smoothed.update_status();

        self.smoothed.output()
    }

    /// Get the shared (un-normalized) float value.
    ///
    /// This can be useful to integrate with various plugin APIs.
    pub fn shared_value(&self) -> Arc<AtomicF32> {
        self.internal.shared_value()
    }
}

// ------  U32  -------------------------------------------------------------------------

/// A parameter with a u32 value
pub struct ParamU32 {
    min_value: u32,
    max_value: u32,
    default_value: u32,

    shared: Arc<AtomicU32>,
}

impl ParamU32 {
    /// Activate the parameter and gets its realtime processor part.
    pub fn activate(&self) -> ParamU32Proc {
        ParamU32Proc {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Load the value of this parameter
    pub fn load_value(&self) -> u32 {
        self.shared.load(Ordering::SeqCst)
    }

    /// The default value of the parameter
    pub fn default_value(&self) -> u32 {
        self.default_value
    }

    /// Set the value of this parameter
    pub fn set_value(&mut self, value: u32) {
        self.shared.store(
            value.clamp(self.min_value, self.max_value),
            Ordering::SeqCst,
        );
    }

    /// The minimum value of this parameter
    pub fn min_value(&self) -> u32 {
        self.min_value
    }

    /// The maximum value of this parameter
    pub fn max_value(&self) -> u32 {
        self.max_value
    }
}

impl Clone for ParamU32 {
    fn clone(&self) -> Self {
        Self {
            min_value: self.min_value,
            max_value: self.max_value,
            default_value: self.default_value,
            shared: Arc::clone(&self.shared),
        }
    }
}

/// The processor for a parameter with a `u32` value.
pub struct ParamU32Proc {
    shared: Arc<AtomicU32>,
}

impl ParamU32Proc {
    /// The (un-normalized) value of this parameter.
    ///
    /// Note, this internally contains an atomic load operation, so prefer to only call this
    /// once per process cycle.
    pub fn load_value(&self) -> u32 {
        self.shared.load(Ordering::SeqCst)
    }

    /// Get the shared u32 value.
    ///
    /// This can be useful to integrate with various plugin APIs.
    pub fn shared_value(&self) -> Arc<AtomicU32> {
        Arc::clone(&self.shared)
    }
}

// ------  bool ------------------------------------------------------------------------

/// A parameter with a boolean value
pub struct ParamBool {
    shared: Arc<AtomicBool>,
    default_value: bool,
}

impl ParamBool {
    /// Activate the parameter and gets its realtime processor part.
    pub fn activate(&self) -> ParamBoolProc {
        ParamBoolProc {
            shared: Arc::clone(&self.shared),
        }
    }

    /// Load the boolean value of this parameter
    pub fn load_value(&self) -> bool {
        self.shared.load(Ordering::SeqCst)
    }

    /// The default boolean value of the parameter
    pub fn default_value(&self) -> bool {
        self.default_value
    }

    /// Set the boolean value of this parameter
    pub fn set_value(&mut self, value: bool) {
        self.shared.store(value, Ordering::SeqCst);
    }
}

impl Clone for ParamBool {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
            default_value: self.default_value,
        }
    }
}

/// The processor for a parameter with an `bool` value
pub struct ParamBoolProc {
    shared: Arc<AtomicBool>,
}

impl ParamBoolProc {
    /// Load the boolean value of this parameter.
    ///
    /// Note, this internally contains an atomic load operation, so prefer to only call this
    /// once per process cycle.
    pub fn load_value(&self) -> bool {
        self.shared.load(Ordering::SeqCst)
    }

    /// Get the shared boolean value.
    ///
    /// This can be useful to integrate with various plugin APIs.
    pub fn shared_value(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.shared)
    }
}
