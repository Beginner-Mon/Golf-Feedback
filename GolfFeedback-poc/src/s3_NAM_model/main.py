import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from nam.config import defaults
from nam.data import FoldedDataset
from nam.data import NAMDataset
from nam.models import NAM
from nam.models import get_num_units
from nam.trainer import LitNAM
from nam.types import Config
from nam.utils import parse_args
from nam.utils import plot_mean_feature_importance
from nam.utils import plot_nams
import matplotlib.pyplot as plt
from plot_figure import plot_nam_feature
def get_config():
    args = parse_args()

    config = defaults()
    config.update(**vars(args))

    return config


def main():
    config = get_config()
    pl.seed_everything(config.seed)

    print(config)
    # exit()

    if config.cross_val:
        dataset = FoldedDataset(
            config,
            data_path=config.data_path,
            features_columns=["income_2", "WP1219", "WP1220", "year", "weo_gdpc_con_ppp"],
            targets_column="WP16",
            weights_column="wgt",
        )
        dataloaders = dataset.train_dataloaders()

        model = NAM(
            config=config,
            name=config.experiment_name,
            num_inputs=len(dataset[0][0]),
            num_units=get_num_units(config, dataset.features),
        )

        for fold, (trainloader, valloader) in enumerate(dataloaders):

            # Folder hack
            tb_logger = TensorBoardLogger(save_dir=config.logdir, name=f'{model.name}', version=f'fold_{fold + 1}')

            checkpoint_callback = ModelCheckpoint(filename=tb_logger.log_dir + "/{epoch:02d}-{val_loss:.4f}",
                                                  monitor='val_loss',
                                                  save_top_k=config.save_top_k,
                                                  mode='min')

            litmodel = LitNAM(config, model)
            trainer = pl.Trainer(logger=tb_logger,
                                 max_epochs=config.num_epochs,
                                 checkpoint_callback=checkpoint_callback)
            trainer.fit(litmodel, train_dataloader=trainloader, val_dataloaders=valloader)

            plot_mean_feature_importance(litmodel.model, dataset)
            plot_nams(litmodel.model, dataset, num_cols=1)
            plt.show()

    else:
        dataset = NAMDataset(
            config,
            data_path=config.data_path,
            features_columns=config.features_columns,
            targets_column=config.targets_column,
         
        )
        trainloader, valloader = dataset.train_dataloaders()
        testloader = dataset.test_dataloaders()

        model = NAM(
            config=config,
            name=config.experiment_name,
            num_inputs=len(dataset[0][0]),
            num_units=get_num_units(config, dataset.features),
        )
        
        # Folder hack
        tb_logger = TensorBoardLogger(save_dir=config.logdir, name=f'{model.name}', version=f'0')

        checkpoint_callback = ModelCheckpoint(filename="{epoch:02d}-{val_loss:.4f}",
                                              monitor='val_loss',
                                              save_top_k=config.save_top_k,
                                              mode='min')

        litmodel = LitNAM(config, model)
        trainer = pl.Trainer(logger=tb_logger, max_epochs=config.num_epochs, callbacks=[checkpoint_callback])
        trainer.fit(litmodel, trainloader, valloader)

        test_results = trainer.test(litmodel, testloader, ckpt_path='best', verbose=True)
        plot_mean_feature_importance(litmodel.model, dataset)
        plot_nams(litmodel.model, dataset, num_cols=1)
        plot_nam_feature(litmodel.model, dataset, feature_name=config.features_columns[1], target='maximize')



if __name__ == "__main__":
# python main.py --data_path faceon_cleaned.csv --targets_column BallSpeed --features_columns 0-STANCE-RATIO 0-UPPER-TILT 3-HIP-SHIFTED 4-HEAD-LOC --regression True --shuffle True --use_dnn True --num_epochs 10 --learning_rate 0.001 --experiment_name BS --dataset_name faceon
# python main.py --data_path faceon_cleaned.csv --targets_column DirectionAngle_binary --features_columns 0-STANCE-RATIO 1-HEAD-LOC 2-HEAD-LOC 2-SHOULDER-LOC  --shuffle True --use_dnn True --num_epochs 10 --learning_rate 0.001 --experiment_name DA --dataset_name faceon
    main()
