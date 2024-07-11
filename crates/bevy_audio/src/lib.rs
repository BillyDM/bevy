#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![forbid(unsafe_code)]
#![doc(
    html_logo_url = "https://bevyengine.org/assets/icon.png",
    html_favicon_url = "https://bevyengine.org/assets/icon.png"
)]

//! Audio playback and processing for bevy.
//!
//! This is to be used with a system audio backend like `bevy_cpal`.

pub mod param;
pub mod util;

/// Commonly used types for audio
pub mod prelude {
    pub use crate::param::{NormalVal, ParamBool, ParamF32, ParamU32};
    pub use crate::{AudioPlugin, AudioServer};
}

use std::num::NonZeroUsize;

use bevy_app::prelude::*;
use bevy_utils::{Duration, Instant};

use self::param::{Curve, NormalVal, ParamF32, SmoothedParamF32Proc, Unit, DEFAULT_SMOOTH_SECS};

/// The default maximum buffer size
pub const DEFAULT_MAX_BLOCK_SIZE: usize = 256;

/// Add support for audio playback and processing to a Bevy Application
pub struct AudioPlugin {
    /// The global volume
    pub global_volume: NormalVal,
}

impl Default for AudioPlugin {
    fn default() -> Self {
        Self {
            global_volume: NormalVal::ONE,
        }
    }
}

impl Plugin for AudioPlugin {
    fn build(&self, app: &mut App) {
        app.insert_non_send_resource(AudioServer::new(self.global_volume));
    }
}

/// A resource containing the audio server.
pub struct AudioServer {
    /// The global output volume
    pub global_volume: ParamF32,

    active_info: Option<ActiveServerInfo>,
}

impl AudioServer {
    /// Create a new audio server
    pub fn new(global_volume: NormalVal) -> Self {
        Self {
            global_volume: ParamF32::from_normalized(
                global_volume,
                0.0,
                -100.0,
                0.0,
                Curve::from_gain_db(-100.0, 0.0),
                Unit::Decibels,
            ),
            active_info: None,
        }
    }

    /// Activate the audio server with the given parameters.
    pub fn activate(&mut self, info: ActiveServerInfo) -> AudioServerProcessor {
        self.active_info = Some(info);

        AudioServerProcessor {
            max_block_size: info.max_block_size,
            global_volume: self.global_volume.activate_with_smoother(
                info.sample_rate,
                DEFAULT_SMOOTH_SECS,
                info.max_block_size,
            ),
            input_channels: vec![vec![0.0; info.max_block_size]; info.num_in_channels],
            output_channels: vec![vec![0.0; info.max_block_size]; info.num_out_channels.into()],
        }
    }

    /// Notify the server that the audio processor counterpart has been dropped.
    pub fn on_deactivated(&mut self) {
        self.active_info = None;
    }

    /// Returns whether or not the server is currently active and processing audio.
    pub fn is_active(&self) -> bool {
        self.active_info.is_some()
    }

    /// Returns information about the activated server.
    pub fn active_info(&self) -> &Option<ActiveServerInfo> {
        &self.active_info
    }
}

/// Information about an active audio server
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveServerInfo {
    /// The sample rate of the stream
    pub sample_rate: u32,
    /// The maximum block size
    pub max_block_size: usize,
    /// The number of input channels
    pub num_in_channels: usize,
    /// The number of output channels
    pub num_out_channels: NonZeroUsize,
}

/// The audio-thread part of the audio server.
pub struct AudioServerProcessor {
    max_block_size: usize,

    global_volume: SmoothedParamF32Proc,

    input_channels: Vec<Vec<f32>>,
    output_channels: Vec<Vec<f32>>,
}

impl AudioServerProcessor {
    /// Process the given buffers.
    pub fn process_interleaved(
        &mut self,
        input_buffer: &[f32],
        output_buffer: &mut [f32],
        info: &StreamCallbackInfo,
    ) {
        let frames = output_buffer.len() / self.output_channels.len();

        // Process in blocks
        let mut frames_processed = 0;
        while frames_processed < frames {
            let block_frames = (frames - frames_processed).min(self.max_block_size);

            if !self.input_channels.is_empty() {
                crate::util::deinterleave(
                    &input_buffer[frames_processed * self.input_channels.len()
                        ..(frames_processed + block_frames) * self.input_channels.len()],
                    &mut self.input_channels,
                );
            }

            self.process_block(frames, info);

            crate::util::interleave(
                &self.output_channels,
                &mut output_buffer[frames_processed * self.output_channels.len()
                    ..(frames_processed + frames) * self.output_channels.len()],
            );

            frames_processed += frames;
        }
    }

    fn process_block(&mut self, frames: usize, _info: &StreamCallbackInfo) {
        let input_silence_mask = SilenceMask::from_channels_slow(&self.input_channels, frames);

        for b in self.output_channels.iter_mut() {
            b[0..frames].fill(0.0);
        }
        let mut output_silence_mask = SilenceMask::new_all_silent(self.output_channels.len());

        // ... Do processing stuff ...

        // Apply the global gain to the output
        let gain_smoothed = self.global_volume.smoothed(frames);
        if !output_silence_mask.all_channels_silent_fast(&self.output_channels) {
            for ch in self.output_channels.iter_mut() {
                for (s, amp) in &mut ch[0..frames]
                    .iter_mut()
                    .zip(&gain_smoothed.values[0..frames])
                {
                    *s *= *amp;
                }
            }
        }
    }
}

/// Additional info about an audio stream for the [`AudioServerProcessor::process`] callback.
pub struct StreamCallbackInfo {
    /// The instant the output data callback was invoked.
    pub callback_timestamp: Instant,

    /// The estimated time between [`StreamCallbackInfo::callback_timestamp`] and the
    /// instant the data will be delivered to the playback device.
    pub output_latency: Duration,
    // TODO
    // /// The estimated time between when the data was read from the input device and
    // /// [`StreamCallbackInfo::callback_timestamp`].
    // pub input_latency: Duration,
}

/// A mask which specifies which channels contain silence.
///
/// This can be used for optimization by skipping processing for inputs
/// that contain silence.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SilenceMask(pub u32);

impl SilenceMask {
    /// Create a new silence mask from the channels by thouroughly
    /// checking every sample.
    pub fn from_channels_slow(channels: &[Vec<f32>], frames: usize) -> Self {
        if !channels.is_empty() {
            let mut mask: u32 = 0;

            for (ch_i, ch) in channels.iter().enumerate() {
                let mut is_silent = true;

                for val in &ch[0..frames] {
                    if *val != 0.0 {
                        is_silent = false;
                        break;
                    }
                }

                if is_silent {
                    mask |= 1 << ch_i;
                }
            }

            Self(mask)
        } else {
            Self(0)
        }
    }

    /// Create a new silence mask with all flags set.
    pub fn new_all_silent(num_channels: usize) -> Self {
        if num_channels == 0 {
            Self(0)
        } else {
            Self((1 << num_channels) - 1)
        }
    }

    /// Returns whether or not all flags are set for all channels.
    pub fn all_channels_silent_fast(&self, channels: &[Vec<f32>]) -> bool {
        if channels.is_empty() {
            true
        } else {
            let num_channels = channels.len();
            let all_silent_mask = (1 << num_channels) - 1;
            self.0 & all_silent_mask == all_silent_mask
        }
    }

    /// Returns whether or not the silent flag is set for a given channel.
    #[inline]
    pub fn is_channel_silent_fast(&self, index: usize) -> bool {
        self.0 & (1 << index) != 0
    }
}
