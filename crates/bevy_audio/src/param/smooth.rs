// Some modified code from baseplug:
//
// https://github.com/wrl/baseplug/blob/trunk/src/smooth.rs
// https://github.com/wrl/baseplug/blob/trunk/LICENSE-APACHE
// https://github.com/wrl/baseplug/blob/trunk/LICENSE-MIT

use std::fmt;
use std::ops;
use std::slice;

const SETTLE: f32 = 0.00001f32;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum SmoothStatus {
    Inactive,
    Active,
    Deactivating,
}

impl SmoothStatus {
    fn is_active(&self) -> bool {
        self != &SmoothStatus::Inactive
    }
}

/// The output of a smoothed parameter
pub struct SmoothOutput<'a> {
    /// The buffer of smoothed values
    pub values: &'a [f32],
    /// The current status of smoothing
    pub status: SmoothStatus,
}

impl<'a> SmoothOutput<'a> {
    /// Returns `true` if the buffer is currently smoothing
    pub fn is_smoothing(&self) -> bool {
        self.status.is_active()
    }
}

impl<'a, I> ops::Index<I> for SmoothOutput<'a>
where
    I: slice::SliceIndex<[f32]>,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, idx: I) -> &I::Output {
        &self.values[idx]
    }
}

/// A parameter smoother
pub(super) struct Smooth {
    output: Vec<f32>,
    input: f32,

    status: SmoothStatus,

    a: f32,
    b: f32,
    last_output: f32,
}

impl Smooth {
    pub fn new(input: f32, max_blocksize: usize) -> Self {
        Self {
            status: SmoothStatus::Inactive,
            input,
            output: vec![input; max_blocksize],

            a: 1.0,
            b: 0.0,
            last_output: input,
        }
    }

    pub fn reset(&mut self, val: f32) {
        self.status = SmoothStatus::Inactive;
        self.input = val;
        self.last_output = val;

        let max_blocksize = self.output.len();

        self.output.clear();
        self.output.resize(max_blocksize, val);
    }

    pub fn set(&mut self, val: f32) {
        self.input = val;
        self.status = SmoothStatus::Active;
    }

    pub fn output(&self) -> SmoothOutput {
        SmoothOutput {
            values: &self.output,
            status: self.status,
        }
    }

    pub fn update_status_with_epsilon(&mut self, epsilon: f32) -> SmoothStatus {
        let status = self.status;

        match status {
            SmoothStatus::Active => {
                if (self.input - self.output[0]).abs() < epsilon {
                    self.reset(self.input);
                    self.status = SmoothStatus::Deactivating;
                }
            }

            SmoothStatus::Deactivating => self.status = SmoothStatus::Inactive,

            _ => (),
        };

        self.status
    }

    pub fn process(&mut self, frames: usize) {
        if self.status != SmoothStatus::Active || frames == 0 {
            return;
        }

        let frames = frames.min(self.output.len());
        let input = self.input * self.a;

        self.output[0] = input + (self.last_output * self.b);

        for i in 1..frames {
            self.output[i] = input + (self.output[i - 1] * self.b);
        }

        self.last_output = self.output[frames - 1];
    }

    pub fn set_speed(&mut self, sample_rate: u32, seconds: f32) {
        self.b = (-1.0f32 / (seconds * sample_rate as f32)).exp();
        self.a = 1.0f32 - self.b;
    }

    pub fn update_status(&mut self) -> SmoothStatus {
        self.update_status_with_epsilon(SETTLE)
    }
}

impl fmt::Debug for Smooth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(concat!("Smooth"))
            .field("output[0]", &self.output[0])
            .field("max_blocksize", &self.output.len())
            .field("input", &self.input)
            .field("status", &self.status)
            .field("last_output", &self.last_output)
            .finish()
    }
}
