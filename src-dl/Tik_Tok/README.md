[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tik-tok-the-utility-of-packet-timing-in/website-fingerprinting-attacks-on-website)](https://paperswithcode.com/sota/website-fingerprinting-attacks-on-website?p=tik-tok-the-utility-of-packet-timing-in)

:warning: :warning: :warning: Experimental - **PLEASE BE CAREFUL**. Intended for Reasearch purposes ONLY. :warning: :warning: :warning:

## The following code was obtained from: [https://github.com/msrocean/Tik_Tok](https://github.com/msrocean/Tik_Tok), and have gotten some small changes.  

This repository contains the code and data to demonstrate the ***Experiments*** and ***Reproduce*** the results of the **Privacy Enhancing Technologies Symposium (PETS) 2020** paper:
### Tik-Tok: The Utility of Packet Timing in Website Fingerprinting Attacks ([Read the Paper](https://petsymposium.org/popets/2020/popets-2020-0043.pdf))


#### Reference Format
```
@article{rahman2020tik,
  title={{Tik-Tok}: The utility of packet timing in website fingerprinting attacks},
  author={Rahman, Mohammad Saidur and Sirinam, Payap and Mathews, Nate and Gangadhara, Kantha Girish and Wright, Matthew},
  journal={Proceedings on Privacy Enhancing Technologies},
  volume={2020},
  number={3},
  pages={5--24},
  year={2020},
  publisher={Sciendo}
}
```

### Reproducability of the Results

#### Dependencies & Required Packages
Please make sure you have all the dependencies available and installed before running the models.
- NVIDIA GPU should be installed in the machine, running on CPU will significantly increase time complexity.
- Ubuntu 16.04.5
- Python3-venv
- Keras version: 2.3.0
- TensorFlow version: 1.14.0
- CUDA Version: 10.2 
- CuDNN Version: 7 
- Python Version: 3.6.x 

Please install the required packages using:

```angular2
pip3 install -r requirements.txt
```

We explain the ways to reproduce each of 
experimental results one by one as the following:

#### 1. Main Part: Timing Features 
  ```angular2
  python Tik_Tok_timing_features.py
  ```

  A snippet of output for your dataset data:

      
      python Tik_Tok_timing_features.py

      Using TensorFlow backend.
      76000 train samples
      9500 validation samples
      9500 test samples
      Train on 76000 samples, validate on 9500 samples
      Epoch 1/100
       - 11s - loss: 4.1017 - acc: 0.0593 - val_loss: 2.9626 - val_acc: 0.1926
      Epoch 2/100
       - 7s - loss: 2.9497 - acc: 0.1976 - val_loss: 2.4673 - val_acc: 0.3026

      .....
  
         Epoch 99/100
           - 7s - loss: 0.3103 - acc: 0.9109 - val_loss: 0.7414 - val_acc: 0.8216
          Epoch 100/100
           - 7s - loss: 0.3096 - acc: 0.9104 - val_loss: 0.7639 - val_acc: 0.8239
    
          Testing accuracy: 0.843284285
          ```

#### 2. Optional: Closed and Open-world Experiments w/ Deep Fingerprinting

See the `DL_Experiments` directory for the scripts used to perform the Direction, Raw Timing, and Directional Timing experiments.

### Questions, Comments, & Feedback
Please, address any questions, comments, or feedback to the authors of the paper.
The main developers of this code are:
 
* Mohammad Saidur Rahman ([saidur.rahman@mail.rit.edu](mailto:saidur.rahman@mail.rit.edu)) 
* Nate Mathews ([nate.mathews@mail.rit.edu](mailto:nate.mathews@mail.rit.edu))
* Payap Sirinam ([payap_siri@rtaf.mi.th](mailto:payap_siri@rtaf.mi.th))
* Kantha Girish Gangadhara ([kantha.gangadhara@mail.rit.edu](mailto:kantha.gangadhara@mail.rit.edu))
* Matthew Wright ([matthew.wright@rit.edu](mailto:matthew.wright@rit.edu))


### Acknowledgements
We thank the anonymous reviewers for their helpful feedback. We give special thanks to Tao Wang for providing details about the technical implementation of the W-T defense, and to Marc Juarez for providing guidelines on developing the W-T prototype. This material is based upon work supported in part by the **National Science Foundation (NSF)** under Grants No. **1722743** and **1816851**.
