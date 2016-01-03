extern crate stft;
use stft::{STFT, WindowType};

#[test]
fn test_log10_positive() {
    assert!(stft::log10_positive(-1. as f64).is_nan());
    assert_eq!(stft::log10_positive(0.), 0.);
    assert_eq!(stft::log10_positive(1.), 0.);
    assert_eq!(stft::log10_positive(10.), 1.);
    assert_eq!(stft::log10_positive(100.), 2.);
    assert_eq!(stft::log10_positive(1000.), 3.);
}

#[test]
fn test_stft() {
    let mut stft = STFT::<f64>::new(WindowType::Hanning, 8, 4);
    assert!(!stft.can_compute());
    assert_eq!(stft.output_size(), 4);
    assert_eq!(stft.len(), 0);
    stft.feed(&vec![500., 0., 100.][..]);
    assert_eq!(stft.len(), 3);
    assert!(!stft.can_compute());
    stft.feed(&vec![500., 0., 100., 0.][..]);
    assert_eq!(stft.len(), 7);
    assert!(!stft.can_compute());

    stft.feed(&vec![500.][..]);
    assert!(stft.can_compute());

    let mut output: Vec<f64> = vec![0.; 4];
    stft.compute(&mut output[..]);
    println!("{:?}", output);
}
