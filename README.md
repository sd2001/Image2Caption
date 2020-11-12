# <div align="center"> 🤖Auto Caption Generation for Images📸
<p align='center'> 
 <img src="https://img.shields.io/badge/-Auto Image2Caption-brightgreen?style=for-the-badge" />
 <img src="https://forthebadge.com/images/badges/built-with-love.svg" />
 <img src="https://img.shields.io/badge/-By%20Swarnabha%20Das-yellow?style=for-the-badge" /><br>
 <img src="http://ForTheBadge.com/images/badges/made-with-python.svg"/>
</p>
</div>

**Image Captioning is the process of generating textual description of an image. It uses both *Natural Language Processing and Computer Vision* to generate the captions. Deep Learning using *CNNs-LSTMs* can be used to solve this problem of generating a caption for a given image, hence called Image Captioning.**

## <div align="center"> ⌛Output that we get💻
<p align='center'> 
 <img src="https://github.com/sd2001/Auto-Image2Caption/blob/main/correct1.png" />
 <img src="https://github.com/sd2001/Auto-Image2Caption/blob/main/correct2.png" /> 
</p>
</div> 

### <div align="center"> There's a lot of biasing as well, since training data wasn't big enough!</div>

<p align='center'> 
 <img src="https://github.com/sd2001/Auto-Image2Caption/blob/main/bias1.png" />
 <img src="https://github.com/sd2001/Auto-Image2Caption/blob/main/bias2.png" /> 
</p>
 
 ## <div align="center"> 🖐️LETS TAKE A QUICK DIVE INTO THIS MAGIC!😇</div>

## Dataset:
- Flickr 8k (containing 8k images), 
- Flickr 30k (containing 30k images), 
- MS COCO (containing 180k images), etc.

**Point to Note:**
> Here I have used the Flickr8k dataset based on the availability of standard computational resources. This dataset is the best for 8GB RAM, and takes about 25mins/epoch training on a CPU. 
  Flickr30k and MS COCO may need about 32GB-64GB RAM based on how it's processed. Consider using AWS EC2 workstation for the best and fastest output. Its paid tho😞!
  
## <div align="center">General Architecture
<p align='center'> 
 <img src="https://github.com/sd2001/Auto-Image2Caption/blob/main/model.png" /> 
</p>
  </div>

## <div align="center"> Model Architecture(VGG16 + LSTMs)
<p align='center'> 
 <img src="https://github.com/sd2001/Auto-Image2Caption/blob/main/vgg16.png" /></p>
</div>

### <div align="center"> We remove the last 2 layers of VGG16 and pass it to 👇 
 
 <p align='center'> 
 <img src="https://github.com/sd2001/Auto-Image2Caption/blob/main/lstm.png" />
 <img src="https://github.com/sd2001/Auto-Image2Caption/blob/main/summary.png" />
</p>
</div>

## <div align="center"> 📊Data that we feed into the Network!📁
  <br><p align='center'>
  <img src="https://github.com/sd2001/Auto-Image2Caption/blob/main/encoded.png" />
  <img src="https://github.com/sd2001/Auto-Image2Caption/blob/main/flow.png" />
  </p>  
 </div>
 
 
 
