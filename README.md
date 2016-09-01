This repository contains our implementation of:

Generating Videos with Scene Dynamics
Carl Vondrick, Hamed Pirsiavash, Antonio Torralba
NIPS 2016

To train a generator for video, see main.lua. For the conditional version, see
main_conditional.lua. To generate videos, see generate.lua

The data loading is designed assuming videos have been stabilized and flattened
into JPEG images. For our stabilization code, see the extra directory.
Essentially, you should convert each video into an image of vertically
concatenated frames (this is to make IO more efficient). You then pass a text
file of these frames into the learning code.

We provide some pre-trained generators in models/

The code is based on DCGAN in Torch7.

If you find this useful for your research, please consider citing our NIPS
paper.
