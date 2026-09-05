use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::thread;
use thiserror::Error;
use wasapi::{initialize_mta, DeviceEnumerator, Direction, SampleType, StreamMode};

const FRAMES_PER_CHUNK: usize = 1_600;

#[derive(Debug, Error)]
pub enum MicrophoneError {
    #[error("failed to start microphone: {0}")]
    Start(String),
}

/// Owns a background WASAPI capture thread and exposes mono 16-bit PCM frames.
pub struct MicrophoneInput {
    frames: Receiver<Vec<i16>>,
    sample_rate: u32,
}

impl MicrophoneInput {
    pub fn start_default() -> Result<Self, MicrophoneError> {
        let (frame_sender, frames) = mpsc::sync_channel(16);
        let (startup_sender, startup_receiver) = mpsc::sync_channel(1);

        thread::Builder::new()
            .name("wakeword-microphone".to_string())
            .spawn(move || capture_loop(frame_sender, startup_sender))
            .map_err(|error| MicrophoneError::Start(error.to_string()))?;

        let sample_rate = startup_receiver
            .recv()
            .map_err(|error| MicrophoneError::Start(error.to_string()))?
            .map_err(MicrophoneError::Start)?;

        Ok(Self {
            frames,
            sample_rate,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn next_frame(&self) -> Option<Vec<i16>> {
        self.frames.recv().ok()
    }

    /// Discards audio captured while the application was waiting for input.
    pub fn clear_pending(&self) {
        while self.frames.try_recv().is_ok() {}
    }
}

fn capture_loop(
    frame_sender: SyncSender<Vec<i16>>,
    startup_sender: SyncSender<Result<u32, String>>,
) {
    let result = run_capture(frame_sender, &startup_sender);
    if let Err(error) = result {
        let _ = startup_sender.try_send(Err(error.clone()));
        log::error!("Microphone capture stopped: {error}");
    }
}

fn run_capture(
    frame_sender: SyncSender<Vec<i16>>,
    startup_sender: &SyncSender<Result<u32, String>>,
) -> Result<(), String> {
    initialize_mta().ok().map_err(|error| error.to_string())?;
    let enumerator = DeviceEnumerator::new().map_err(|error| error.to_string())?;
    let device = enumerator
        .get_default_device(&Direction::Capture)
        .map_err(|error| error.to_string())?;
    let mut audio_client = device
        .get_iaudioclient()
        .map_err(|error| error.to_string())?;

    // The Windows mix format is guaranteed to be accepted by the device.
    let format = audio_client
        .get_mixformat()
        .map_err(|error| error.to_string())?;
    let sample_rate = format.get_samplespersec();
    let channels = format.get_nchannels() as usize;
    let bytes_per_frame = format.get_blockalign() as usize;
    let bits_per_sample = format.get_bitspersample();
    let sample_type = format.get_subformat().map_err(|error| error.to_string())?;
    let mode = StreamMode::PollingShared {
        autoconvert: false,
        buffer_duration_hns: 0,
    };

    audio_client
        .initialize_client(&format, &Direction::Capture, &mode)
        .map_err(|error| {
            format!("initialize WASAPI client: {error}; format={format:?}; mode={mode:?}")
        })?;
    let capture_client = audio_client
        .get_audiocaptureclient()
        .map_err(|error| error.to_string())?;
    audio_client
        .start_stream()
        .map_err(|error| error.to_string())?;
    startup_sender
        .send(Ok(sample_rate))
        .map_err(|error| error.to_string())?;

    let mut bytes = VecDeque::new();
    loop {
        capture_client
            .read_from_device_to_deque(&mut bytes)
            .map_err(|error| error.to_string())?;
        thread::sleep(std::time::Duration::from_millis(10));

        while bytes.len() >= FRAMES_PER_CHUNK * bytes_per_frame {
            let mut frame = Vec::with_capacity(FRAMES_PER_CHUNK);
            for _ in 0..FRAMES_PER_CHUNK {
                let raw: Vec<u8> = (0..bytes_per_frame)
                    .map(|_| bytes.pop_front().unwrap())
                    .collect();
                frame.push(decode_first_channel(
                    &raw,
                    channels,
                    bits_per_sample,
                    &sample_type,
                )?);
            }

            if frame_sender.send(frame).is_err() {
                audio_client.stop_stream().ok();
                return Ok(());
            }
        }
    }
}

fn decode_first_channel(
    frame: &[u8],
    channels: usize,
    bits_per_sample: u16,
    sample_type: &SampleType,
) -> Result<i16, String> {
    let bytes_per_sample = frame.len() / channels;
    let sample = &frame[..bytes_per_sample];

    match (sample_type, bits_per_sample) {
        (SampleType::Float, 32) => {
            let bytes: [u8; 4] = sample.try_into().map_err(|_| "invalid f32 sample")?;
            Ok((f32::from_le_bytes(bytes).clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        }
        (SampleType::Int, 16) => {
            let bytes: [u8; 2] = sample.try_into().map_err(|_| "invalid i16 sample")?;
            Ok(i16::from_le_bytes(bytes))
        }
        (SampleType::Int, 24) => {
            let value =
                ((sample[2] as i32) << 24 | (sample[1] as i32) << 16 | (sample[0] as i32) << 8)
                    >> 16;
            Ok((value >> 8) as i16)
        }
        (SampleType::Int, 32) => {
            let bytes: [u8; 4] = sample.try_into().map_err(|_| "invalid i32 sample")?;
            Ok((i32::from_le_bytes(bytes) >> 16) as i16)
        }
        _ => Err(format!(
            "unsupported microphone format: {sample_type:?} {bits_per_sample}-bit"
        )),
    }
}
