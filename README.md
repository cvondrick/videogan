This repository contains our implementation of:

Generating Videos with Scene Dynamics
Carl Vondrick, Hamed Pirsiavash, Antonio Torralba
NIPS 2016

To train a generator on video, see main.lua.

For the conditional version, see main_conditional.lua. 

To generate videos, see generate.lua

The data loading is designed assuming videos have been stabilized
and flattened into JPEG images. For our stabilization code see,
see the extra directory. Essentially, you should convert a video
into a vertically concatenated frames.

We provide some pre-trained models in models/

The code is based on DCGAN in Torch7.
