import soundfile as sf
import numpy as np
import tensorflow as tf

# The sampling rate of 16k is fixed.
block_len = 512
block_shift = 128
# load model
model = tf.saved_model.load('./pretrained_model/final_model_after_training')
infer = model.signatures["serving_default"]
# load audio file at 16k fs (please change)
audio,fs = sf.read('./sample_audio/Unp_Front_SSN_135.wav')
# check for sampling rate
if fs != 16000:
    raise ValueError('This model only supports 16k sampling rate. Provided : ' + fs )
# preallocate output audio
out_file = np.zeros((len(audio)))
# create buffer
in_buffer = np.zeros((block_len))
out_buffer = np.zeros((block_len))
# calculate number of blocks
num_blocks = (audio.shape[0] - (block_len-block_shift)) // block_shift
# iterate over the number of blcoks        
for idx in range(num_blocks):
    # shift values and write to buffer
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = audio[idx*block_shift:(idx*block_shift)+block_shift]
    # create a batch dimension of one
    in_block = np.expand_dims(in_buffer, axis=0).astype('float32')
    # process one block
    out_block= infer(tf.constant(in_block))['conv1d_1']
    # shift values and write to buffer
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = np.zeros((block_shift))
    out_buffer  += np.squeeze(out_block)
    # write block to output file
    out_file[idx*block_shift:(idx*block_shift)+block_shift] = out_buffer[:block_shift]
    
    
# write to .wav file 
sf.write('out.wav', out_file, fs) 

print('Processing finished.')
