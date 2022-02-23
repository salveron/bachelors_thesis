# What I have so far for the practical part of the thesis:


### 1. Basic knowledge about digital signal processing:

- work with .wav files, channels (left/right);
- counting samplerate/duration/samples;
- producing simple sounds:
  - making pure tones, white noise, clicks;
  - adding them together, concatenating them
- drawing sound waves, FFT outputs, spectrograms.


### 2. Knowledge about how to apply filters:

- gammatone *(first attempt: I drew an impulse response and the filtered sound wave)*;
- gammatone filter bank *(cochleagram)*;
- Hann filter *("failed" attempt - need to investigate better - I was probably applying it on my
time frames (rectangular windows) incorrectly)*;
- low-pass/high-pass filters.


### 3. Basic knowledge about the CASA systems:

#### Main stages that I need to implement for my CASA system:
- peripheral analysis
- feature extraction  *(<- now I'm somewhere here)*
- segmentation
- grouping
- resynthesis

#### Overall:
I was studying CASA systems meant mainly for the problem of speech recognition, and I haven't found anything
good for music specifically. Pretty much all the sources that I had so far were implementing
algorithms for separating "target sounds" from the "background", and I couldn't really imagine
how to apply it normally on polyphonic sounds.

Having in mind that I needed to produce "good" outputs from the CASA system to pass over to an algorithm
for Automatic Music Transcription ("good" meant "clear" for me back then), I started to realize that
my initial idea would probably fail, because in my observations I was finding that the outputs of common
CASA systems were not "100% separated" (the background was rather attenuated, and not fully masked). Another
complicating factor was that I needed to work with polyphonic sounds, and if they weren't masked fully,
then (I was assuming that) the AMT algorithm would not give the clearest transcription for the tones from
different streams.

I was also trying to simplify some steps. For example, after computing the cochleagram for the peripheral
analysis stage, I was thinking about how to extract F0 (fundamental frequency) features from the T-F units.
I tried to split the sound to small windows and then, seeing that the data had a lot of "spikes", I
computed RMS values (along time axis for every time window) and applied gaussian filtering (along frequency
axis for every frequency channel). This allowed me to find F0 by finding the biggest amplitude across
the frequency channels, but it was working poorly in places where piano sounds started to be less and less
distinguishable from noise. For polyphony, this approach was not easy at all, because I needed to
find two maximums lying in different spikes (which sometimes weren't maximums at all, because of some
destructive interference of two target frequencies).

Having failed with simplifying, I started to compute a correlogram for my data: I computed autocorrelations
for every frequency channel for different values of lags, summary autocorrelation across all channels to see
when the harmonics align, and cross-channel correlation for neighboring frequency channels to see which
T-F units are in agreement with each other (neighboring channels are sometimes giving similar results,
because the frequencies near the target frequency of the channel filter are not usually much attenuated).
This will help me to do segmentation and grouping based on F0 cues.

The most common goal for a CASA system is to find an IBM (ideal binary mask) for the cochleagram to
mask the background and emphasize the target sound. I was experimenting with masks too, but haven't achieved
good results. As an idea for a good experiment, I want to try to work with binary masks (only 0 or 1 values)
as well as with probability masks (every value from the [0, 1] interval). It would be interesting to
compare results and see whether the outputs are somehow different.

#### Which sounds I've been experimenting with:

- **Alternating and gallop:** from the "Auditory neuroscience" book I've read that when humans listen to
these kinds of sounds, the brain is sometimes processing them as a single "stream", and sometimes as two
different "streams". These recognition patterns are also changing in time, and depend on the difference 
between the fundamental frequencies of the tones, and tempo. These sounds are called non-simultaneous,
but they are not really polyphonic.
- **Simultaneous:** this C5-A3 sound had two notes played simultaneously and was my primary input
for experiments with polyphony (the attempts were described above).
- **Monophony:** this recording was made after it was decided to use monophonic sounds first and see the
results. I also added some artificial noise to see how it would affect the output. Now this is my primary
input for experiments, and it looks like the most promising one.