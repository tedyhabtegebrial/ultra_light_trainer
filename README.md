## ultra_light_trainer
Ultra Light Multi-GPU trainer


```
# Preffered option
# if you are not using git with ssh replace the repo with git@github.com:tedyhabtegebrial/ultra_light_trainer.git 
pip3 install --user git+https://github.com/tedyhabtegebrial/ultra_light_trainer.git

# Option 2:

git clone https://github.com/tedyhabtegebrial/ultra_light_trainer.git
cd ultra_light_trainer
sudo python3 setup.py develop


```


```python

## Training a model
## Check example folder for more details
log_dir = "./logs/exp_001"
tb_logger = TBLogger(logdir=os.path.join(log_dir, "tensor_board"))
topk_logger = TopKLogger(topk=2,
                        logdir=os.path.join(log_dir, "topk"),
                        monitor="val/psnr",
                        monitor_mode="max")
ult_model = UlMLP()
trainer = UlTrainer(
    model=ult_model,
    topk_logger=topk_logger,
    tb_logger=tb_logger,
    validate_every_x_epoch=1,
    max_epochs=40,
    log_root=log_dir,
    tb_log_steps=2)


trainer.train()

```