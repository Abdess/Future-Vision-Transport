import matplotlib.pyplot as plt
import time
# Générateur et augmentation des données
from cityscapesscripts.preparation.batchdatagenaug import BatchDataGenAug
from cityscapesscripts.helpers.visualization_utils import \
    print_segmentation_onto_image, create_video_from_images
from math import ceil

import cityscapes_tensorflow
# Variable d'exportation des 8 catégories principale
from cityscapesscripts.helpers.labels_bdga import IDS_TO_CATEGORYIDS_ARRAY, \
    COLORS_CAT_TO_IDS_DICT
from train_config import num_classes, train_batch_size, val_batch_size
from train_config import train_images, val_images, test_images, train_gt, val_gt
from train_config import vgg_pretrained, epochs

# Mettez les chemins vers les jeux de données dans des listes, car c'est ce
# que `BatchDataGenAug` demande en entrée.

train_image_dirs = [train_images]
train_ground_truth_dirs = [train_gt]
val_image_dirs = [val_images]
val_ground_truth_dirs = [val_gt]
model_seg = FCN # SegNet ou UNet

train_dataset = BatchDataGenAug(image_dirs=train_image_dirs,
                               image_file_extension='png',
                               ground_truth_dirs=train_ground_truth_dirs,
                               image_name_split_separator='leftImg8bit',
                               ground_truth_suffix='gtFine_labelIds',
                               check_existence=True,
                               num_classes=num_classes)

val_dataset = BatchDataGenAug(image_dirs=val_image_dirs,
                             image_file_extension='png',
                             ground_truth_dirs=val_ground_truth_dirs,
                             image_name_split_separator='leftImg8bit',
                             ground_truth_suffix='gtFine_labelIds',
                             check_existence=True,
                             num_classes=num_classes)

num_train_images = train_dataset.get_num_files()
num_val_images = val_dataset.get_num_files()

print("Taille de l'ensemble de données d'entraînement : ", num_train_images, " images")
print("Taille de l'ensemble de données de validation : ", num_val_images, " images")

# ----------------------------------------------------------------------------
#                   Générateur de données
# ----------------------------------------------------------------------------

# Réglage de la même taille de lot pour les deux générateurs ici.

train_generator = train_dataset.generate(batch_size=train_batch_size,
                                         convert_colors_to_ids=False,
                                         convert_ids_to_ids=False,
                                         convert_to_one_hot=True,
                                         void_class_id=None,
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         brightness=False,
                                         flip=0.5,
                                         translate=False,
                                         scale=False,
                                         gray=False,
                                         to_disk=False,
                                         shuffle=True)

val_generator = val_dataset.generate(batch_size=val_batch_size,
                                     convert_colors_to_ids=False,
                                     convert_ids_to_ids=False,
                                     convert_to_one_hot=True,
                                     void_class_id=None,
                                     random_crop=False,
                                     crop=False,
                                     resize=False,
                                     brightness=False,
                                     flip=False,
                                     translate=False,
                                     scale=False,
                                     gray=False,
                                     to_disk=False,
                                     shuffle=True)

# Afficher quelques diagnostics pour s'assurer que nos lots ne sont pas vides
# et qu'il ne faut pas un temps infini pour les générer.

start_time = time.time()
images, gt_images = next(train_generator)
print('Temps de généreration d\'un lot : {:.3f} secondes'.format(time.time() -
                                                          start_time))
print('Nombre d\'images générées :', len(images))
print('Nombre d\'images de vérité de terrain générées :', len(gt_images))

# Visualiser l'ensemble des données
# Générer des lots à partir du générateur train_generator où le ground truth
# n'est pas converti en one-hot pour que nous puissions les représenter
# sous forme d'images.

example_generator = train_dataset.generate(batch_size=train_batch_size,
                                           convert_to_one_hot=False)

# Générer un lot, et visualiser.

example_images, example_gt_images = next(example_generator)
i = 0  # Sélectionner l'échantillon du lot à afficher ci-dessous.

figure, cells = plt.subplots(1, 2, figsize=(16, 8))
cells[0].imshow(example_images[i])
cells[1].imshow(example_gt_images[i])
plt.show()

# ----------------------------------------------------------------------------
#                   Fin du générateur de données
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
#                   Créer le modèle pour l'entraînement
# ----------------------------------------------------------------------------
model = model_seg(model_load_dir=None,
                  tags=None,
                  vgg16_dir=vgg_pretrained,
                  num_classes=num_classes,
                  variables_load_dir=None)


# Définir une fonction de planification du taux d'apprentissage à passer
# à la méthode `train()`.
def learning_rate_schedule(step):
    if step <= 10000:
        return 0.0001
    elif 10000 < step <= 20000:
        return 0.00001
    elif 20000 < step <= 40000:
        return 0.000003
    else:
        return 0.000001


model.train(train_generator=train_generator,
            epochs=epochs,
            steps_per_epoch=ceil(num_train_images / train_batch_size),
            learning_rate_schedule=learning_rate_schedule,
            keep_prob=0.5,
            l2_regularization=0.0,
            eval_dataset='val',
            eval_frequency=2,
            val_generator=val_generator,
            val_steps=ceil(num_val_images / val_batch_size),
            metrics=['loss', 'mean_iou', 'accuracy'],
            save_during_training=True,
            save_dir='cityscapes_model',
            save_best_only=True,
            save_tags=['default'],
            save_name='(batch-size-4)',
            save_frequency=2,
            saver='saved_model',
            monitor='loss',
            record_summaries=True,
            summaries_frequency=10,
            summaries_dir='./tensorboard_log/cityscapes',
            summaries_name='configuration_01',
            training_loss_display_averaging=3)

model.save(model_save_dir='cityscapes_model',
           saver='saved_model',
           tags=['default'],
           name='(batch-size-4)',
           include_global_step=True,
           include_last_training_loss=True,
           include_metrics=True,
           force_save=False)

model.evaluate(data_generator=val_generator,
               metrics=['loss', 'mean_iou', 'accuracy'],
               num_batches=ceil(num_val_images / val_batch_size),
               l2_regularization=0.0,
               dataset='val')
