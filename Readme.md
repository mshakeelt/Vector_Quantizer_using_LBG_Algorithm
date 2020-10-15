We have implemented a vector quantizer for audio signals using LBG machine learning algorithm. We have Used two audio signals here one for training and other for testing

# In "scalar_Quantizer.py" we are 	
* Reading two audio files and applying a mid-tread quantizer with 16 levels (corresponding to 4 bits)
* We are saving the original file and quantized files in binary format

We are Implementing a vector quantizer (VQ) with dimension N=2 and M=16^2=256 code vectors (again corresponding to 4 bits per dimension)

#Training stage (training.py)
* We are Training the VQ using the LBG algorithm on the training audio dataset
* Then we are Ploting codebook (red stars) with 2D training signal 1b (blue dots) and “voronoi regions” (green)
* We are saving our training set to “codebook.bin” file and “voronoi regions” to “voronoi_regions.bin”

# Encoding stage (Vector_Quantizer_Encoder.py)
* We are Ploting codebook (red stars) with 2D testing audio signal (blue dots) and “voronoi regions” (green)
* We are Encoding the testing signal with training set from “codebook.bin” and Saving indices to “coded_vq_signal.bin”

# Decoding stage (Vector_Quantizer_Decoder.py)
* We are decoding (reconstructing) the signal from “coded_vq_signal.bin”
* And finally we are ploting the decoded (de-quantized) signal together with original