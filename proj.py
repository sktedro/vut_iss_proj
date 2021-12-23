from scipy.io import wavfile
from scipy import signal
from math import e
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import wave
import struct

# TODO v cos ani sin nemá byť 2 * pi asi 

def normalizeData(inputSignal, sampleCount):
    # Subtract the average
    avgVal = sum(inputSignal) / sampleCount
    normInputAudio = inputSignal - avgVal

    # Divide by the maximum value
    normInputAudio = normInputAudio / max(abs(normInputAudio))

    return normInputAudio

def getFrames(inputSignal, sampleCount):
    # Prepare the frames
    frames = []
    for i in range(sampleCount // 512 - 1):
        frames.append(inputSignal[i * 512 : i * 512 + 1024])
    return frames

def dft(x, N):
    # X[k] = sum from 0 to N-1 of (x[n] * e^(-jkn2pi / N))
    coeffs = []
    e = np.exp(-1j * 2 * np.pi / N)
    for k in range(0, N // 2):
        coeffs.append(sum(x * [e ** (k * n) for n in range(N)]))
    return coeffs
    
def transformSignal(frames, sampleRate):
    print("Calculating the DFT")

    # FFT
    magnitudes = abs(np.array([np.fft.fft(frames[k])[0: 1024 // 2] for k in range(len(frames))]))

    # DFT
    #  coeffs = np.array([dft(frames[frame], 1024) for frame in range(len(frames))])
    #  magnitudes = abs(coeffs)

    frequencies = [k * sampleRate // 1024 for k in range(1024 // 2)]
    print("DFT calculated")

    return magnitudes, frequencies

def plotDft(frequencies, magnitudes):
    plt.plot(frequencies, magnitudes[0])
    plt.ylabel("Magnitude []")
    plt.xlabel("Frequency [Hz]")
    plt.show()

def getSpectrogram(magnitudes):
    return np.transpose(10 * np.log10(abs(magnitudes) ** 2))

def plotSpectrogram(frequencies, spectrogram, audioLength):
    time = np.arange(0, audioLength, audioLength / len(frames))
    plt.pcolormesh(time, frequencies, spectrogram, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.show()

def getDisruptiveFreqs(cosAmount, magnitudes, frequencies):
    # Get average magnitudes for all of the frames
    magnitudesTransposed = np.transpose(magnitudes)
    magnitudesAvg = np.array([sum(magnitudesTransposed[freq]) for freq in range(len(magnitudesTransposed))])

    # Get indices of 4 frequencies with the maximum magnitudes
    maxMagnitudesIndices = np.argpartition(magnitudesAvg, -cosAmount)[-cosAmount:]

    # Get the frequencies of these indices
    disruptiveFreqs = []
    for i in range(cosAmount):
        disruptiveFreqs.append(frequencies[maxMagnitudesIndices[i]])
    print("Approx disruptive frequencies: " + str(disruptiveFreqs))
    return disruptiveFreqs

def getDisruptiveCosines(cosAmount, disruptiveFreqs, sampleCount, sampleRate):
    cosX = np.arange(0, sampleCount) 
    cosY = []
    for i in range(cosAmount):
        cosY.append([np.cos(2 * np.pi * disruptiveFreqs[i] * t / sampleRate) for t in cosX])
    return cosX, cosY

def generateDisruptiveCosines(cosAmount, cosY):
    cosYTransposed = np.transpose(cosY)
    disruptiveCosines = []
    for i in range(len(cosY[0])):
        disruptiveCosines.append(sum(cosYTransposed[i]) / cosAmount)
    return disruptiveCosines

def writeDisruptiveCosines(cos, sampleCount, sampleRate):
    wav_file=wave.open("./audio/4cos.wav", "w")
    wav_file.setparams((1, 2, sampleRate, sampleCount, "NONE", ""))
    for sample in cos:
        wav_file.writeframes(struct.pack('h', int(sample * sampleRate)))

def createZPKFilters(disruptiveFreqs, filterAmount, sampleRate):
    print("Filters: ")
    filterLowerBound = (disruptiveFreqs - 15)
    filterLowerPass = (disruptiveFreqs - 50)
    filterUpperBound = (disruptiveFreqs + 15)
    filterUpperPass = (disruptiveFreqs + 50) 
    zeros = []
    poles = []
    gains = []
    for i in range(filterAmount):
        wp = [filterLowerPass[i], filterUpperPass[i]]
        ws = [filterLowerBound[i], filterUpperBound[i]]
        order = signal.buttord(wp, ws, 3, 40, fs = sampleRate)
        print(order)
        z, p, k = signal.butter(order[0], order[1], "bandstop", False, "zpk", sampleRate)
        zeros.append(z)
        poles.append(p)
        gains.append(k)
    return zeros, poles, gains

def plotZP(zeros, poles):
    circle1 = plt.Circle((0, 0), 1, fill = False)
    circle2 = plt.Circle((0, 0), 1, fill = False)
    plt.gca().add_patch(circle1)
    plt.scatter(zeros.real, zeros.imag, color = 'g')
    plt.title("Zeros")
    plt.axis([-1, 1, -1, 1])
    plt.show()
    plt.gca().add_patch(circle2)
    plt.scatter(poles.real, poles.imag, color = 'b')
    plt.title("Poles")
    plt.axis([-1, 1, -1, 1])
    plt.show()

def plotFilterImpulseResponse(sos, freqs, plotCount, sampleCount):
    impulseResponseInputs = np.zeros(sampleCount)
    impulseResponseInputs[0] = 1
    fig, axes = plt.subplots(plotCount)
    for i in range(len(sos)):
        impulseResponse = signal.sosfilt(sos[i], impulseResponseInputs)
        axes[i].set_title("Filter for frequency " + str(freqs[i]) + " Hz")
        axes[i].plot(impulseResponse)
    fig.tight_layout()
    plt.show()

def plotFilterFrequencyResponses(sos, sampleCount, sampleRate):
    fig, axes = plt.subplots(len(sos))
    for i in range(len(sos)):
        freqs, mags = signal.sosfreqz(sos[i], sampleCount, False, sampleRate)
        axes[i].plot(freqs, abs(mags))
    plt.show()


print("==================================================")


## Read the input audio file

sampleRate, inputSignal = wavfile.read('./audio/xskalo01.wav')
print("Sample rate: ", sampleRate)

# Get the input audio file length
sampleCount = len(inputSignal)
print("Amount of samples: ", sampleCount)
audioLength = sampleCount / sampleRate
print("Audio length [s]: ", audioLength)

# Plot the input signal
#  plt.plot(np.arange(0, audioLength, audioLength / sampleCount)[0: -1], inputSignal)
#  plt.show()

# Normalize the data
normInputAudio = normalizeData(inputSignal, sampleCount)

# Get frames
frames = getFrames(normInputAudio, sampleCount)
print("We have", len(frames), "frames, all with length", len(frames[0]))

# Plot one of the frames
#  plt.plot(np.arange(0, 1024 / sampleRate, 1 / sampleRate), frames[24])
#  plt.show()


## Discrete fourier transform

magnitudes, frequencies = transformSignal(frames, sampleRate)

# Plot the result
#  plotDft(frequencies, magnitudes)


## Spectrogram

# Calculate the spectrogram values
spectrogram = getSpectrogram(magnitudes)

# Plot the spectrogram
#  plotSpectrogram(frequencies, spectrogram, audioLength)


## Finding the disruptive frequencies

# We know there will be four of them
disruptiveFreqsAmount = 4

# Get frequencies of the disruptive cosines
disruptiveFreqs = np.array(getDisruptiveFreqs(disruptiveFreqsAmount, magnitudes, frequencies))

# Generate the disruptive cosines with the frequencies detected
disruptiveCosinesX, disruptiveCosinesY = getDisruptiveCosines(disruptiveFreqsAmount, disruptiveFreqs, sampleCount, sampleRate)
disruptiveCosines = generateDisruptiveCosines(disruptiveFreqsAmount, disruptiveCosinesY)

# Write the cosines to a file
writeDisruptiveCosines(disruptiveCosines, sampleCount, sampleRate)


## Creating a filter

# Create the ZPK filters
zeros, poles, gains = createZPKFilters(disruptiveFreqs, disruptiveFreqsAmount, sampleRate)

# Plot the zeros and poles of the filter
#  print("Zeros: ")
#  print(zeros[0])
#  print("Poles: ")
#  print(poles[0])
#  plotZP(np.array(zeros[0]), np.array(poles[0]))

# Convert ZPK to SOS so it's easier to work with
sos = [signal.zpk2sos(zeros[i], poles[i], gains[i]) for i in range(disruptiveFreqsAmount)]

# Calculate and plot the impulse responses
#  plotFilterImpulseResponse(sos, disruptiveFreqs, disruptiveFreqsAmount, 1024)

# Calculate and plot the frequency response
#  plotFilterFrequencyResponses(sos, sampleCount, sampleRate)


## Filter the signal, normalize it and write it to a file

# Filter the audio by all four filters, one by one
filteredAudio = normInputAudio
for filterSetting in sos:
    filteredAudio = signal.sosfilt(filterSetting, filteredAudio)

# Normalize the filtered audio
filteredAudio = normalizeData(filteredAudio, sampleCount)

# Plot a spectrogram of the filtered audio
#  frames = getFrames(filteredAudio, sampleCount)
#  magnitudes, frequencies = transformSignal(frames, sampleRate)
#  spectrogram = getSpectrogram(magnitudes)
#  plotSpectrogram(frequencies, spectrogram, audioLength)

# Write the audio to a file
wav_file = wave.open("./audio/clean_bandstop.wav", "w")
wav_file.setparams((1, 2, sampleRate, sampleCount, "NONE", ""))
for sample in filteredAudio:
    wav_file.writeframes(struct.pack('h', int(sample * 0x7fff)))




#  quit(0)
### BONUSES

## Get the exact frequencies
def getExactDisruptiveFreqs(inputSignal, sampleRate):
    sampleCount = len(inputSignal)
    frames = getFrames(inputSignal, sampleCount)
    magnitudes = abs(np.array([np.fft.fft(frames[k])[0: 1024 // 2] for k in range(len(frames))]))
    frequencies = [k * sampleRate // 1024 for k in range(1024)]
    approxFreqs = getDisruptiveFreqs(4, magnitudes, frequencies)

    exactFreqs = []
    maxCorrelCoeffs = []

    # Get the exact disruptive frequencies
    for freq in approxFreqs:
        # Search in interval <freq - 20, freq + 20>
        testRange = 8
        testFreqs = np.arange(freq - testRange, freq + testRange, 1)
        correlCoeffs = []
        testSamples = int(sampleCount / 10)

        # For every frequency in that inteval, generate a cosine and 
        # multiply the input signal by it to get correlation coeffs
        for testFreq in testFreqs:
            # TODO which one? With 2pi?
            cos = [np.cos(2 * np.pi * testFreq * n / sampleRate) for n in range(testSamples)]
            #  cos = [np.cos(testFreq * n / sampleRate) for n in range(testSamples)]
            correlCoeffs.append(int(abs(sum(inputSignal[0: testSamples] * cos))))

        # Get the frequency with the highest correlation coefficient
        maxCorrelCoeff = max(np.array(correlCoeffs))
        correlCoeffIndex = correlCoeffs.index(maxCorrelCoeff)
        maxCorrelCoeffs.append(maxCorrelCoeff)
        exactFreqs.append(testFreqs[correlCoeffIndex])

    return exactFreqs, maxCorrelCoeffs
#  disruptiveFreqs, maxCorrelCoeffs = getExactDisruptiveFreqs(inputSignal, sampleRate)
#  print("Exact disruptive frequencies: " + str(disruptiveFreqs))


#  c = 16384
#  magnitudes = abs(np.fft.fft(inputSignal[0: c])[0: c // 2])
#  frequencies = [k * sampleRate // c for k in range(c)]

print("Calculating DFT")
magnitudes = abs(dft(inputSignal, sampleCount))
print("DFT calculated")
frequencies = [k * sampleRate // sampleCount for k in range(sampleCount)]

maxMagnitudesIndices = np.argpartition(magnitudes, -4)[-4:]
disruptiveFreqs = []
for i in range(4):
    disruptiveFreqs.append(frequencies[maxMagnitudesIndices[i]])
print("Exact disruptive frequencies: " + str(disruptiveFreqs))

#  print("Their correlation coefficients: ")
#  print(maxCorrelCoeffs)

def getDisruptiveFreqsAmplitudes(inputSignal, framesAmount, sampleCount):
    sampleCount = len(inputSignal)
    frames = getFrames(inputSignal, sampleCount)[0: framesAmount]
    magnitudes = abs(np.array([np.fft.fft(frames[k])[0: 1024 // 2] for k in range(len(frames))]))
    approxFreqs = getDisruptiveFreqs(4, magnitudes)

    for freq in approxFreqs:
        amplitude = []
        for magnitude in magnitudes:
            amplitude.append(magnitude[freq * 1024 // sampleRate])
        amplitude = (sum(amplitude) / len(amplitude)) * 2 / 1024
        print("Amplitude of a cosine with frequency " + str(freq) + " is " + str(amplitude))

#  getDisruptiveFreqsAmplitudes(inputSignal, disruptiveFreqsAmount, sampleCount)
