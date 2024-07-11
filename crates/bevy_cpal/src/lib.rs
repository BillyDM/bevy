#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![forbid(unsafe_code)]
#![doc(
    html_logo_url = "https://bevyengine.org/assets/icon.png",
    html_favicon_url = "https://bevyengine.org/assets/icon.png"
)]

//! A [`cpal`] backend for `bevy_audio`

use std::num::NonZeroUsize;

use bevy_app::prelude::*;
use bevy_audio::ActiveServerInfo;
use bevy_utils::{
    tracing::{debug, error, warn},
    Duration, Instant,
};
use cpal::{
    traits::{DeviceTrait, HostTrait},
    SupportedBufferSize,
};
use ringbuf::traits::Split;

const STREAM_TIMEOUT: Duration = Duration::from_secs(10);

/// The configuration for creating a [`cpal`] audio stream.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpalConfig {
    /// If `true`, then no input audio stream will be created.
    ///
    /// Defaults to `true`
    pub output_only: bool,
    /// The host to use for the input audio device. This has no effect if
    /// [`CpalConfig::output_only`] is set to `true`.
    ///
    /// Set to `None` to select the system's default host.
    ///
    /// Defaults to `None`
    pub input_host: Option<cpal::HostId>,
    /// The name of the input audio device to use. This has no effect if
    /// [`CpalConfig::output_only`] is set to `true`.
    ///
    /// If the audio device isn't found, then the system's default input device
    /// will be used instead.
    ///
    /// Set to `None` to select the system's default input device.
    ///
    /// Defaults to `None`
    pub input_device_name: Option<String>,
    /// The host to use for the output audio device.
    ///
    /// Set to `None` to select the system's default host.
    ///
    /// Defaults to `None`
    pub output_host: Option<cpal::HostId>,
    /// The name of the output audio device to use.
    ///
    /// If the audio device isn't found, then the system's default output device
    /// will be used instead.
    ///
    /// Set to `None` to select the system's default output device.
    ///
    /// Defaults to `None`
    pub output_device_name: Option<String>,
    /// The sample rate to use.
    ///
    /// Set to `None` to use the system's default sample rate.
    ///
    /// Note that the resulting stream may fallback to a different sample rate if the
    /// requested one isn't supported.
    ///
    /// Defaults to `None`
    pub sample_rate: Option<u32>,
    /// The maximum buffer size to use for processing. Different values may result
    /// in better performance. (This should be typically somewhere between 16 and 2048,
    /// and ideally be a power of two).
    ///
    /// Default to `256`
    pub max_block_size: usize,
    /// The latency between the input and output audio streams.
    ///
    /// If the value is too low, then underruns may occur which will result in
    /// input data being lost.
    ///
    /// Defaults to `150 ms`
    pub io_sync_latency: Duration,
}

impl Default for CpalConfig {
    fn default() -> Self {
        Self {
            output_only: true,
            input_host: None,
            input_device_name: None,
            output_host: None,
            output_device_name: None,
            sample_rate: None,
            max_block_size: bevy_audio::DEFAULT_MAX_BLOCK_SIZE,
            io_sync_latency: Duration::from_millis(150),
        }
    }
}

/// Add support for audio playback and processing using the [`cpal`] backend to a Bevy Application
#[derive(Default)]
pub struct CpalPlugin {
    /// The configuration for creating a [`cpal`] stream.
    pub config: CpalConfig,
}

impl Plugin for CpalPlugin {
    fn name(&self) -> &str {
        "bevy_cpal::CpalPlugin"
    }

    fn build(&self, app: &mut App) {
        app.add_plugins(bevy_audio::AudioPlugin::default());

        let (input_stream, input_stream_info) = if !self.config.output_only {
            if let Some((stream, info)) = open_input_stream(&self.config) {
                (Some(stream), Some(info))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let output_stream = {
            let mut audio_server = app
                .world_mut()
                .get_non_send_resource_mut::<bevy_audio::AudioServer>()
                .unwrap();

            open_output_stream(&self.config, audio_server.as_mut(), input_stream_info)
        };

        app.insert_non_send_resource(CpalPluginResource {
            _input_stream: input_stream,
            _output_stream: output_stream,
        });
    }
}

/// A resource containing the CPAL stream and the audio server.
pub struct CpalPluginResource {
    _input_stream: Option<cpal::Stream>,
    _output_stream: Option<cpal::Stream>,
}

struct InputStreamInfo {
    sample_rate: u32,
    channels: usize,
    sync_buffer_rx: ringbuf::HeapCons<f32>,
}

fn open_input_stream(cpal_config: &CpalConfig) -> Option<(cpal::Stream, InputStreamInfo)> {
    use ringbuf::traits::Producer;

    const SYNC_BUFFER_SIZE_MULTIPLIER: f64 = 4.0;

    let host = if let Some(host_id) = cpal_config.input_host {
        match cpal::host_from_id(host_id) {
            Ok(host) => host,
            Err(e) => {
                error!("Could not find audio host {host_id:?}: {e}. Falling back to default host");
                cpal::default_host()
            }
        }
    } else {
        cpal::default_host()
    };

    let maybe_device = if let Some(name) = &cpal_config.input_device_name {
        match host.output_devices() {
            Ok(mut devices) => {
                let found_device = devices.find(|d| d.name().map(|n| &n == name).unwrap_or(false));

                if found_device.is_none() {
                    warn!("Could not find input audio device with name {name}. Falling back to default input device");
                    host.default_input_device()
                } else {
                    found_device
                }
            }
            Err(e) => {
                error!("Could not get list of audio input devices: {e}. Falling back to default input device");
                host.default_input_device()
            }
        }
    } else {
        host.default_input_device()
    };

    let Some(device) = maybe_device else {
        error!("No input audio device found. Audio input will be disabled");
        return None;
    };

    let maybe_config = if let Some(sample_rate) = cpal_config.sample_rate {
        match device.supported_input_configs() {
            Ok(mut configs) => {
                let found_config = configs.find(|c| {
                    c.try_with_sample_rate(cpal::SampleRate(sample_rate))
                        .is_some()
                });

                if let Some(config_range) = found_config {
                    Some(config_range.with_sample_rate(cpal::SampleRate(sample_rate)))
                } else {
                    warn!("Could not find supported config with sample rate {sample_rate} for input audio device. Falling back to a default config");
                    match device.default_output_config() {
                        Ok(c) => Some(c),
                        Err(e) => {
                            error!("Could not get default config for input audio device: {e}. Audio input will be disabled");
                            None
                        }
                    }
                }
            }
            Err(e) => {
                error!("Could not get supported configs for input audio device: {e}. Audio input will be disabled");
                None
            }
        }
    } else {
        match device.default_output_config() {
            Ok(c) => Some(c),
            Err(e) => {
                error!("Could not get default config for input audio device: {e}. Audio input will be disabled");
                None
            }
        }
    };

    let Some(stream_config) = maybe_config else {
        return None;
    };

    let sample_rate = stream_config.sample_rate().0;
    let channels = stream_config.channels();
    let sync_buffer_size = (cpal_config.io_sync_latency.as_secs_f64()
        * sample_rate as f64
        * stream_config.channels() as f64
        * SYNC_BUFFER_SIZE_MULTIPLIER)
        .ceil() as usize;

    let (mut sync_buffer_tx, sync_buffer_rx) =
        ringbuf::HeapRb::<f32>::new(sync_buffer_size).split();

    match device.build_input_stream(
        &stream_config.config(),
        move |data: &[f32], _info| {
            let num_pushed = sync_buffer_tx.push_slice(data);

            if num_pushed < data.len() {
                debug!("Audio input underrun");
            }
        },
        move |e| {
            // TODO: Send a message that the audio stream was closed.
            error!("An error occurred on input audio stream: {e}");
        },
        Some(STREAM_TIMEOUT),
    ) {
        Ok(stream) => Some((
            stream,
            InputStreamInfo {
                sample_rate,
                channels: channels as usize,
                sync_buffer_rx,
            },
        )),
        Err(e) => {
            error!(
                "Error occured while opening audio input stream: {e}. Audio input will be disabled"
            );
            None
        }
    }
}

fn open_output_stream(
    cpal_config: &CpalConfig,
    audio_server: &mut bevy_audio::AudioServer,
    mut input_stream_info: Option<InputStreamInfo>,
) -> Option<cpal::Stream> {
    use ringbuf::traits::Consumer;

    const FALLBACK_INPUT_BUFFER_SIZE: Duration = Duration::from_secs(1);

    let host = if let Some(host_id) = cpal_config.output_host {
        match cpal::host_from_id(host_id) {
            Ok(host) => host,
            Err(e) => {
                error!("Could not find audio host {host_id:?}: {e}. Falling back to default host");
                cpal::default_host()
            }
        }
    } else {
        cpal::default_host()
    };

    let maybe_device = if let Some(name) = &cpal_config.output_device_name {
        match host.output_devices() {
            Ok(mut devices) => {
                let found_device = devices.find(|d| d.name().map(|n| &n == name).unwrap_or(false));

                if found_device.is_none() {
                    warn!("Could not find output audio device with name {name}. Falling back to default output device");
                    host.default_output_device()
                } else {
                    found_device
                }
            }
            Err(e) => {
                error!("Could not get list of audio output devices: {e}. Falling back to default output device");
                host.default_output_device()
            }
        }
    } else {
        host.default_output_device()
    };

    let Some(device) = maybe_device else {
        error!("No output audio device found. Audio output will be disabled");
        return None;
    };

    let maybe_config = if let Some(sample_rate) = cpal_config.sample_rate {
        match device.supported_output_configs() {
            Ok(mut configs) => {
                let found_config = configs.find(|c| {
                    c.try_with_sample_rate(cpal::SampleRate(sample_rate))
                        .is_some()
                });

                if let Some(config_range) = found_config {
                    Some(config_range.with_sample_rate(cpal::SampleRate(sample_rate)))
                } else {
                    warn!("Could not find supported config with sample rate {sample_rate} for output audio device. Falling back to a default config");
                    match device.default_output_config() {
                        Ok(c) => Some(c),
                        Err(e) => {
                            error!("Could not get default config for output audio device: {e}. Audio output will be disabled");
                            None
                        }
                    }
                }
            }
            Err(e) => {
                error!("Could not get supported configs for output audio device: {e}. Audio output will be disabled");
                None
            }
        }
    } else {
        match device.default_output_config() {
            Ok(c) => Some(c),
            Err(e) => {
                error!("Could not get default config for output audio device: {e}. Audio output will be disabled");
                None
            }
        }
    };

    let Some(stream_config) = maybe_config else {
        return None;
    };

    let sample_rate = stream_config.sample_rate().0;
    let sample_rate_recip = (sample_rate as f64).recip();
    let channels = NonZeroUsize::new(stream_config.channels() as usize).unwrap();

    let mut processor = audio_server.activate(ActiveServerInfo {
        sample_rate,
        max_block_size: cpal_config.max_block_size,
        num_in_channels: input_stream_info.as_ref().map(|i| i.channels).unwrap_or(0),
        num_out_channels: channels,
    });

    let mut input_buffer: Vec<f32> = if let Some(info) = &input_stream_info {
        // TODO: Remove this once we support converting sample rates below.
        assert_eq!(sample_rate, info.sample_rate);

        let buffer_size = match stream_config.buffer_size() {
            SupportedBufferSize::Range { min: _, max } => *max as usize,
            SupportedBufferSize::Unknown => {
                (FALLBACK_INPUT_BUFFER_SIZE.as_secs_f64() * sample_rate as f64).ceil() as usize
            }
        };

        vec![0.0; buffer_size]
    } else {
        Vec::new()
    };

    match device.build_output_stream(
        &stream_config.config(),
        move |data: &mut [f32], info| {
            // TODO: See if cpal already zeros out the ouput buffer for us.
            data.fill(0.0);

            let frames = data.len() / channels;

            let input_buffer_slice = if let Some(info) = &mut input_stream_info {
                if info.sample_rate != sample_rate {
                    // TODO: convert sample rate
                }

                let input_samples = frames * info.channels;

                let samples_written = info
                    .sync_buffer_rx
                    .pop_slice(&mut input_buffer[0..input_samples]);

                // If there was an underrun, fill the rest with zeros.
                if samples_written < input_samples {
                    input_buffer[samples_written..input_samples].fill(0.0);
                }

                &input_buffer[0..input_samples]
            } else {
                &[]
            };

            processor.process_interleaved(
                input_buffer_slice,
                data,
                &bevy_audio::StreamCallbackInfo {
                    callback_timestamp: Instant::now(),
                    output_latency: info
                        .timestamp()
                        .playback
                        .duration_since(&info.timestamp().callback)
                        .unwrap_or_else(|| {
                            Duration::from_secs_f64(
                                (data.len() / channels) as f64 * sample_rate_recip,
                            )
                        }),
                },
            );
        },
        move |e| {
            // TODO: Send a message that the audio stream was closed.
            error!("An error occurred on output audio stream: {e}");
        },
        Some(STREAM_TIMEOUT),
    ) {
        Ok(stream) => Some(stream),
        Err(e) => {
            error!("Error occured while opening audio output stream: {e}. Audio output will be disabled");
            None
        }
    }
}
