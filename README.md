## MDGAN &mdash; Simple TensorFlow Implementation [[Paper]](https://arxiv.org/abs/1811.00152)
### : Mixture Density Generative Adversarial Networks

<div align="center">
  <img src="./assets/teaser.png" width=500px height=500px>
</div>

## Usage
```python

fake_img = generator(noise)

real_logit = discriminator(real_img)
fake_logit = discriminator(fake_img)

g_loss = generator_loss(fake_logit)
d_loss = discriminator_loss(real_logit, fake_logit)

```

## Reference
* [MDGAN-Pytorch](https://github.com/haihabi/MD-GAN)


## Author
[Junho Kim](http://bit.ly/jhkim_ai)
