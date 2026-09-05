mod audio;

use audio::MicrophoneInput;
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

const OUTPUT_SAMPLE_RATE: u32 = 16_000;
const RECORDING_SECONDS: usize = 2;
const DEFAULT_SAMPLE_COUNT: usize = 30;
const SENTENCE_STARTERS: &[&str] = &[
    "<wake word> can you",
    "<wake word> remind me",
    "<wake word> add dinner",
    "<wake word> what's on",
    "<wake word> check my",
    "<wake word> turn on",
    "<wake word> tell me",
    "<wake word> schedule",
];
const NEGATIVE_PHRASES: &[&str] = &[
    "hello",
    "good morning",
    "the mail arrived",
    "a cup of coffee",
    "a nearby restaurant",
    "I received an email",
    "can you check my calendar",
    "remind me this afternoon",
    "what is the weather",
    "turn on the bedroom light",
];

#[derive(Clone, Copy)]
enum RecordingMode {
    Isolated,
    SentenceStarters,
    Negatives,
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let mode = match arguments.first().map(String::as_str) {
        Some("sentences") => RecordingMode::SentenceStarters,
        Some("negatives") => RecordingMode::Negatives,
        _ => RecordingMode::Isolated,
    };
    let count_argument = match mode {
        RecordingMode::Isolated => arguments.first(),
        RecordingMode::SentenceStarters | RecordingMode::Negatives => arguments.get(1),
    };
    let sample_count = count_argument
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(DEFAULT_SAMPLE_COUNT);
    let output_directory = output_directory(mode);
    fs::create_dir_all(&output_directory)?;

    println!("Wake Word voice sample recorder");
    println!(
        "Saving {sample_count} private recordings to {}",
        output_directory.display()
    );
    println!("Use natural variations: quick, slow, relaxed, and different distances.\n");

    let microphone = MicrophoneInput::start_default()?;
    for number in 1..=sample_count {
        let prompt = match mode {
            RecordingMode::Isolated => "<wake word>",
            RecordingMode::SentenceStarters => {
                SENTENCE_STARTERS[(number - 1) % SENTENCE_STARTERS.len()]
            }
            RecordingMode::Negatives => {
                NEGATIVE_PHRASES[(number - 1) % NEGATIVE_PHRASES.len()]
            }
        };
        println!(
            "Sample {number}/{sample_count}: press Enter when ready, then say '{prompt}'."
        );
        wait_for_enter()?;
        countdown();
        microphone.clear_pending();
        println!("RECORDING...");

        let required_samples = microphone.sample_rate() as usize * RECORDING_SECONDS;
        let mut samples = Vec::with_capacity(required_samples);
        while samples.len() < required_samples {
            let Some(frame) = microphone.next_frame() else {
                return Err("microphone stopped while recording".into());
            };
            samples.extend(frame);
        }
        samples.truncate(required_samples);

        let samples = resample(&samples, microphone.sample_rate(), OUTPUT_SAMPLE_RATE);
        let file_name = match mode {
            RecordingMode::Negatives => format!("negative_{number:03}.wav"),
            _ => format!("wakeword_{number:03}.wav"),
        };
        let path = output_directory.join(file_name);
        write_wav(&path, &samples)?;
        println!("Saved {}\n", path.display());
    }

    println!("Finished. Recorded {sample_count} samples.");
    Ok(())
}

fn wait_for_enter() -> io::Result<()> {
    let mut input = String::new();
    io::stdin().read_line(&mut input).map(|_| ())
}

fn countdown() {
    for number in (1..=3).rev() {
        println!("{number}...");
        thread::sleep(Duration::from_millis(500));
    }
}

fn resample(input: &[i16], input_rate: u32, output_rate: u32) -> Vec<i16> {
    if input_rate == output_rate {
        return input.to_vec();
    }

    let output_length = input.len() * output_rate as usize / input_rate as usize;
    (0..output_length)
        .map(|index| {
            let source_position = index as f64 * input_rate as f64 / output_rate as f64;
            let left = source_position.floor() as usize;
            let right = (left + 1).min(input.len() - 1);
            let fraction = source_position - left as f64;
            (input[left] as f64 * (1.0 - fraction) + input[right] as f64 * fraction) as i16
        })
        .collect()
}

fn write_wav(path: &Path, samples: &[i16]) -> Result<(), hound::Error> {
    let specification = hound::WavSpec {
        channels: 1,
        sample_rate: OUTPUT_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, specification)?;
    for sample in samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()
}

fn output_directory(mode: RecordingMode) -> PathBuf {
    let folder = match mode {
        RecordingMode::Isolated => "positive",
        RecordingMode::SentenceStarters => "positive-sentences",
        RecordingMode::Negatives => "negative-tests",
    };
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("training-data")
        .join(folder)
}

#[cfg(test)]
mod tests {
    use super::resample;

    #[test]
    fn resamples_48khz_to_16khz_at_the_expected_length() {
        let input = vec![1_i16; 48_000];
        assert_eq!(resample(&input, 48_000, 16_000).len(), 16_000);
    }
}
