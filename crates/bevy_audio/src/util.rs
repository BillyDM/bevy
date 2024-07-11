//! General conversion functions and utilities.

/// Returns the raw amplitude from the given decibel value.
#[inline]
pub fn db_to_amp(db: f32) -> f32 {
    10.0f32.powf(0.05 * db)
}

/// Returns the decibel value from the raw amplitude.
#[inline]
pub fn amp_to_db(amp: f32) -> f32 {
    20.0 * amp.log(10.0)
}

/// Returns the raw amplitude from the given decibel value.
///
/// If `db <= -100.0`, then 0.0 will be returned instead (negative infinity gain).
#[inline]
pub fn db_to_amp_clamped_neg_100_db(db: f32) -> f32 {
    if db <= -100.0 {
        0.0
    } else {
        db_to_amp(db)
    }
}

/// Returns the decibel value from the raw amplitude value.
///
/// If `amp <= 0.00001`, then the minimum of `-100.0` dB will be
/// returned instead (representing negative infinity gain when paired with
/// [`db_to_amp_clamped_neg_100_db`]).
#[inline]
pub fn amp_to_db_clamped_neg_100_db(amp: f32) -> f32 {
    if amp <= 0.00001 {
        -100.0
    } else {
        amp_to_db(amp)
    }
}

/// Efficiently deinterleave audio.
pub fn deinterleave(interleaved: &[f32], channels: &mut [Vec<f32>]) {
    match channels.len() {
        0 => return,
        1 => {
            let min_len = channels[0].len().min(interleaved.len());
            channels[0][0..interleaved.len()].copy_from_slice(&interleaved[0..min_len]);
        }
        2 => {
            // Provide a loop with optimized stereo deinterleaving
            let (ch0, ch1) = channels.split_first_mut().unwrap();
            let ch0 = ch0.as_mut_slice();
            let ch1 = &mut ch1[0][0..ch0.len()];

            for (input, (ch0s, ch1s)) in interleaved
                .chunks_exact(2)
                .zip(ch0.iter_mut().zip(ch1.iter_mut()))
            {
                *ch0s = input[0];
                *ch1s = input[1];
            }
        }
        n => {
            for (ch_i, ch) in channels.iter_mut().enumerate() {
                for (input, output) in interleaved.iter().skip(ch_i).step_by(n).zip(ch.iter_mut()) {
                    *output = *input;
                }
            }
        }
    }
}

/// Efficiently interleave audio.
pub fn interleave(channels: &[Vec<f32>], interleaved: &mut [f32]) {
    match channels.len() {
        0 => return,
        1 => {
            let min_len = channels[0].len().min(interleaved.len());
            interleaved[0..min_len].copy_from_slice(&channels[0][0..min_len]);
        }
        2 => {
            // Provide a loop with optimized stereo interleaving
            let ch0 = channels[0].as_slice();
            let ch1 = &channels[1][0..ch0.len()];

            for (output, (ch0s, ch1s)) in interleaved
                .chunks_exact_mut(2)
                .zip(ch0.iter().zip(ch1.iter()))
            {
                output[0] = *ch0s;
                output[1] = *ch1s;
            }
        }
        n => {
            for (ch_i, ch) in channels.iter().enumerate() {
                for (output, input) in interleaved.iter_mut().skip(ch_i).step_by(n).zip(ch.iter()) {
                    *output = *input;
                }
            }
        }
    }
}
