###
# Pipeline de segmentation qui effectue
# l’entrainement, l’évaluation, la prédiction,
# la visualisation et la sauvegarde d'un modèle
###

from distutils.version import LooseVersion
import tensorflow as tf
import warnings
from tqdm import trange
import sys
import os.path
import scipy.misc
import shutil
from glob import glob
from collections import deque
import numpy as np
import time
from keras.models import *
from keras.layers import *
from keras.optimizers import *

from cityscapesscripts.helpers.tf_variable_summaries import add_variable_summaries
from cityscapesscripts.helpers.visualization_utils import print_segmentation_onto_image, create_split_view


class FCN8:

    def __init__(self,
                 data_format=None,
                 tags=None,
                 num_classes=None,
                 variables_load_dir=None,
                 input_height=224,
                 input_width=224,
                 pretrained=None):

        assert LooseVersion(tf.__version__) >= LooseVersion(
            '2.0'
        ), 'Ce programme nécessite TensorFlow version 2.0 ou plus récente. Vous utilisez {}'.format(
            tf.__version__)
        print('TensorFlow Version : {}'.format(tf.__version__))

        self.variables_load_dir = variables_load_dir
        self.data_format = data_format
        self.tags = tags
        self.vgg16_tag = 'vgg16'
        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.pretrained = pretrained

        self.variables_updated = False
        self.eval_dataset = None

        self.metric_names = []
        self.metric_values = []
        self.best_metric_values = []
        self.metric_value_tensors = []
        self.metric_update_ops = []
        self.training_loss = None
        self.best_training_loss = 99999999.9
        self.pretrained_vgg16 = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

        self.g_step = None

        self.fcn8s_output, self.l2_regularization_rate = self._build_decoder()

        self.labels = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None, None, self.num_classes],
            name='labels_input')
        self.total_loss, self.train_op, self.learning_rate, self.global_step = self._build_optimizer(
        )

        self.softmax_output, self.predictions_argmax = self._build_predictor()

        self.mean_loss_value, self.mean_loss_update_op, self.mean_iou_value, self.mean_iou_update_op, self.acc_value, self.acc_update_op, self.metrics_reset_op = self._build_metrics(
        )

        self.summaries_training, self.summaries_evaluation = self._build_summary_ops(
        )

        if not variables_load_dir is None:
            checkpoint = tf.train.Checkpoint()
            manager = tf.train.CheckpointManager(checkpoint,
                                                 directory=variables_load_dir,
                                                 max_to_keep=5)
            status = checkpoint.restore(manager.latest_checkpoint)
            while True:
                # train
                manager.save()

    def _build_decoder(self):

        stddev_1x1 = 0.001
        stddev_conv2d_trans = 0.01

        l2_regularization_rate = tf.placeholder(dtype=tf.float32,
                                                shape=[],
                                                name='l2_regularization_rate')

        with tf.name_scope('decoder'):

            pool3_out_scaled = tf.multiply(self.pool3_out,
                                           0.0001,
                                           name='pool3_out_scaled')

            pool3_1x1 = tf.layers.conv2d(
                inputs=pool3_out_scaled,
                filters=self.num_classes,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=stddev_1x1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(
                    l2_regularization_rate),
                name='pool3_1x1')

            pool4_out_scaled = tf.multiply(self.pool4_out,
                                           0.01,
                                           name='pool4_out_scaled')

            pool4_1x1 = tf.layers.conv2d(
                inputs=pool4_out_scaled,
                filters=self.num_classes,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=stddev_1x1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(
                    l2_regularization_rate),
                name='pool4_1x1')

            fc7_1x1 = tf.layers.conv2d(
                inputs=self.fc7_out,
                filters=self.num_classes,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=stddev_1x1),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(
                    l2_regularization_rate),
                name='fc7_1x1')

            fc7_conv2d_trans = tf.layers.conv2d_transpose(
                inputs=fc7_1x1,
                filters=self.num_classes,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=stddev_conv2d_trans),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(
                    l2_regularization_rate),
                name='fc7_conv2d_trans')

            add_fc7_pool4 = tf.add(fc7_conv2d_trans,
                                   pool4_1x1,
                                   name='add_fc7_pool4')

            fc7_pool4_conv2d_trans = tf.layers.conv2d_transpose(
                inputs=add_fc7_pool4,
                filters=self.num_classes,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=stddev_conv2d_trans),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(
                    l2_regularization_rate),
                name='fc7_pool4_conv2d_trans')

            add_fc7_pool4_pool3 = tf.add(fc7_pool4_conv2d_trans,
                                         pool3_1x1,
                                         name='add_fc7_pool4_pool3')

            fc7_pool4_pool3_conv2d_trans = tf.layers.conv2d_transpose(
                inputs=add_fc7_pool4_pool3,
                filters=self.num_classes,
                kernel_size=(16, 16),
                strides=(8, 8),
                padding='same',
                kernel_initializer=tf.truncated_normal_initializer(
                    stddev=stddev_conv2d_trans),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(
                    l2_regularization_rate),
                name='fc7_pool4_pool3_conv2d_trans')

            fcn8s_output = tf.identity(fc7_pool4_pool3_conv2d_trans,
                                       name='fcn8s_output')

        return fc7_pool4_pool3_conv2d_trans, l2_regularization_rate

    def _build_optimizer(self):

        with tf.name_scope('optimizer'):

            global_step = tf.Variable(0, trainable=False, name='global_step')

            learning_rate = tf.placeholder(dtype=tf.float32,
                                           shape=[],
                                           name='learning_rate')

            regularization_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)
            regularization_loss = tf.add_n(regularization_losses,
                                           name='regularization_loss')

            approximation_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.labels, logits=self.fcn8s_output),
                name='approximation_loss')
            total_loss = tf.add(approximation_loss,
                                regularization_loss,
                                name='total_loss')

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                               name='adam_optimizer')
            train_op = optimizer.minimize(total_loss,
                                          global_step=global_step,
                                          name='train_op')

        return total_loss, train_op, learning_rate, global_step

    def _build_predictor(self):

        with tf.name_scope('predictor'):
            softmax_output = tf.nn.softmax(self.fcn8s_output,
                                           name='softmax_output')
            predictions_argmax = tf.argmax(softmax_output,
                                           axis=-1,
                                           name='predictions_argmax',
                                           output_type=tf.int64)

        return softmax_output, predictions_argmax

    def _build_metrics(self):

        with tf.variable_scope('metrics') as scope:
            labels_argmax = tf.argmax(self.labels,
                                      axis=-1,
                                      name='labels_argmax',
                                      output_type=tf.int64)

            mean_loss_value, mean_loss_update_op = tf.metrics.mean(
                self.total_loss)

            mean_loss_value = tf.identity(mean_loss_value,
                                          name='mean_loss_value')
            mean_loss_update_op = tf.identity(mean_loss_update_op,
                                              name='mean_loss_update_op')

            mean_iou_value, mean_iou_update_op = tf.metrics.mean_iou(
                labels=labels_argmax,
                predictions=self.predictions_argmax,
                num_classes=self.num_classes)

            mean_iou_value = tf.identity(mean_iou_value, name='mean_iou_value')
            mean_iou_update_op = tf.identity(mean_iou_update_op,
                                             name='mean_iou_update_op')

            acc_value, acc_update_op = tf.metrics.accuracy(
                labels=labels_argmax, predictions=self.predictions_argmax)

            acc_value = tf.identity(acc_value, name='acc_value')
            acc_update_op = tf.identity(acc_update_op, name='acc_update_op')

            local_metric_vars = tf.contrib.framework.get_variables(
                scope=scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            metrics_reset_op = tf.variables_initializer(
                var_list=local_metric_vars, name='metrics_reset_op')

        return (mean_loss_value, mean_loss_update_op, mean_iou_value,
                mean_iou_update_op, acc_value, acc_update_op, metrics_reset_op)

    def _build_summary_ops(self):
        graph = tf.get_default_graph()

        add_variable_summaries(
            variable=graph.get_tensor_by_name('pool3_1x1/kernel:0'),
            scope='pool3_1x1/kernel')
        add_variable_summaries(
            variable=graph.get_tensor_by_name('pool3_1x1/bias:0'),
            scope='pool3_1x1/bias')
        add_variable_summaries(
            variable=graph.get_tensor_by_name('pool4_1x1/kernel:0'),
            scope='pool4_1x1/kernel')
        add_variable_summaries(
            variable=graph.get_tensor_by_name('pool4_1x1/bias:0'),
            scope='pool4_1x1/bias')
        add_variable_summaries(
            variable=graph.get_tensor_by_name('fc7_1x1/kernel:0'),
            scope='fc7_1x1/kernel')
        add_variable_summaries(
            variable=graph.get_tensor_by_name('fc7_1x1/bias:0'),
            scope='fc7_1x1/bias')
        add_variable_summaries(
            variable=graph.get_tensor_by_name('fc7_conv2d_trans/kernel:0'),
            scope='fc7_conv2d_trans/kernel')
        add_variable_summaries(
            variable=graph.get_tensor_by_name('fc7_conv2d_trans/bias:0'),
            scope='fc7_conv2d_trans/bias')
        add_variable_summaries(variable=graph.get_tensor_by_name(
            'fc7_pool4_conv2d_trans/kernel:0'),
            scope='fc7_pool4_conv2d_trans/kernel')
        add_variable_summaries(
            variable=graph.get_tensor_by_name('fc7_pool4_conv2d_trans/bias:0'),
            scope='fc7_pool4_conv2d_trans/bias')
        add_variable_summaries(variable=graph.get_tensor_by_name(
            'fc7_pool4_pool3_conv2d_trans/kernel:0'),
            scope='fc7_pool4_pool3_conv2d_trans/kernel')
        add_variable_summaries(variable=graph.get_tensor_by_name(
            'fc7_pool4_pool3_conv2d_trans/bias:0'),
            scope='fc7_pool4_pool3_conv2d_trans/bias')
        add_variable_summaries(
            variable=graph.get_tensor_by_name('fc7/weights:0'),
            scope='fc7/kernel')
        add_variable_summaries(
            variable=graph.get_tensor_by_name('fc7/biases:0'),
            scope='fc7/bias')
        add_variable_summaries(
            variable=graph.get_tensor_by_name('fc6/weights:0'),
            scope='fc6/kernel')
        add_variable_summaries(
            variable=graph.get_tensor_by_name('fc6/biases:0'),
            scope='fc6/bias')
        add_variable_summaries(
            variable=graph.get_tensor_by_name('conv4_3/filter:0'),
            scope='conv4_3/kernel')
        add_variable_summaries(
            variable=graph.get_tensor_by_name('conv4_3/biases:0'),
            scope='conv4_3/bias')
        add_variable_summaries(
            variable=graph.get_tensor_by_name('conv3_3/filter:0'),
            scope='conv3_3/kernel')
        add_variable_summaries(
            variable=graph.get_tensor_by_name('conv3_3/biases:0'),
            scope='conv3_3/bias')

        tf.summary.scalar('total_loss', self.total_loss)
        tf.summary.scalar('learning_rate', self.learning_rate)

        summaries_training = tf.summary.merge_all()
        summaries_training = tf.identity(summaries_training,
                                         name='summaries_training')

        mean_loss = tf.summary.scalar('mean_loss', self.mean_loss_value)
        mean_iou = tf.summary.scalar('mean_iou', self.mean_iou_value)
        accuracy = tf.summary.scalar('accuracy', self.acc_value)

        summaries_evaluation = tf.summary.merge(
            inputs=[mean_loss, mean_iou, accuracy])
        summaries_evaluation = tf.identity(summaries_evaluation,
                                           name='summaries_evaluation')

        return summaries_training, summaries_evaluation

    def _initialize_metrics(self, metrics):

        self.metric_names = []
        self.best_metric_values = []
        self.metric_update_ops = []
        self.metric_value_tensors = []

        if 'loss' in metrics:
            self.metric_names.append('loss')
            self.best_metric_values.append(99999999.9)
            self.metric_update_ops.append(self.mean_loss_update_op)
            self.metric_value_tensors.append(self.mean_loss_value)
        if 'mean_iou' in metrics:
            self.metric_names.append('mean_iou')
            self.best_metric_values.append(0.0)
            self.metric_update_ops.append(self.mean_iou_update_op)
            self.metric_value_tensors.append(self.mean_iou_value)
        if 'accuracy' in metrics:
            self.metric_names.append('accuracy')
            self.best_metric_values.append(0.0)
            self.metric_update_ops.append(self.acc_update_op)
            self.metric_value_tensors.append(self.acc_value)

    def train(self,
              train_generator,
              epochs,
              steps_per_epoch,
              learning_rate_schedule,
              keep_prob=0.5,
              l2_regularization=0.0,
              eval_dataset='train',
              eval_frequency=5,
              val_generator=None,
              val_steps=None,
              metrics={},
              save_during_training=False,
              save_dir=None,
              save_best_only=True,
              save_tags=['default'],
              save_name='',
              save_frequency=5,
              saver='saved_model',
              monitor='loss',
              record_summaries=True,
              summaries_frequency=10,
              summaries_dir=None,
              summaries_name=None,
              training_loss_display_averaging=3):

        if not tf.test.gpu_device_name():
            warnings.warn(
                'Aucun GPU trouvé. Veuillez noter que l\'entraînement de ce réseau sera insupportablement lent sans GPU.'
            )
        else:
            print('Périphérique GPU par défaut : {}'.format(
                tf.test.gpu_device_name()))

        if not eval_dataset in ['train', 'val']:
            raise ValueError(
                "`eval_dataset` doit être l'un de 'train' ou 'val', mais est '{}'."
                .format(eval_dataset))

        if (eval_dataset == 'val') and ((val_generator is None) or
                                        (val_steps is None)):
            raise ValueError(
                "Lorsque eval_dataset == 'val', un `val_generator` et `val_steps` doivent être passés."
            )

        for metric in metrics:
            if not metric in ['loss', 'mean_iou', 'accuracy']:
                raise ValueError(
                    "{} n'est pas une métrique valide. Les métriques valides sont ['loss', mean_iou', 'accuracy']."
                    .format(metric))

        if (not monitor in metrics) and (not monitor == 'loss'):
            raise ValueError(
                'Vous essayez de surveiller {}, mais elle n\'est pas dans "metrics" et n\'est donc pas calculée.'
                .format(monitor))

        self.eval_dataset = eval_dataset

        self.g_step = self.sess.run(self.global_step)
        learning_rate = learning_rate_schedule(self.g_step)

        self._initialize_metrics(metrics)

        if record_summaries:
            training_writer = tf.summary.FileWriter(logdir=os.path.join(
                summaries_dir, summaries_name),
                graph=self.sess.graph)
            if len(metrics) > 0:
                evaluation_writer = tf.summary.FileWriter(
                    logdir=os.path.join(summaries_dir, summaries_name +
                                        '_eval'))

        for epoch in range(1, epochs + 1):

            loss_history = deque(maxlen=training_loss_display_averaging)

            tr = trange(steps_per_epoch, file=sys.stdout)
            tr.set_description('Epoch {}/{}'.format(epoch, epochs))

            for train_step in tr:

                batch_images, batch_labels = next(train_generator)

                if record_summaries and (self.g_step % summaries_frequency
                                         == 0):
                    _, current_loss, self.g_step, training_summary = self.sess.run(
                        [
                            self.train_op, self.total_loss, self.global_step,
                            self.summaries_training
                        ],
                        feed_dict={
                            self.image_input: batch_images,
                            self.labels: batch_labels,
                            self.learning_rate: learning_rate,
                            self.keep_prob: keep_prob,
                            self.l2_regularization_rate: l2_regularization
                        })
                    training_writer.add_summary(summary=training_summary,
                                                global_step=self.g_step)
                else:
                    _, current_loss, self.g_step = self.sess.run(
                        [self.train_op, self.total_loss, self.global_step],
                        feed_dict={
                            self.image_input: batch_images,
                            self.labels: batch_labels,
                            self.learning_rate: learning_rate,
                            self.keep_prob: keep_prob,
                            self.l2_regularization_rate: l2_regularization
                        })

                self.variables_updated = True

                loss_history.append(current_loss)
                losses = np.array(loss_history)
                self.training_loss = np.mean(losses)

                tr.set_postfix(ordered_dict={
                    'loss': self.training_loss,
                    'learning rate': learning_rate
                })

                learning_rate = learning_rate_schedule(self.g_step)

            if (len(metrics) > 0) and (epoch % eval_frequency == 0):

                if eval_dataset == 'train':
                    data_generator = train_generator
                    num_batches = steps_per_epoch
                    description = 'Évaluation sur le jeu de données d\'entrainement'
                elif eval_dataset == 'val':
                    data_generator = val_generator
                    num_batches = val_steps
                    description = 'Évaluation sur le jeu de données d\'entrainement'

                self._evaluate(data_generator=data_generator,
                               metrics=metrics,
                               num_batches=num_batches,
                               l2_regularization=l2_regularization,
                               description=description)

                if record_summaries:
                    evaluation_summary = self.sess.run(
                        self.summaries_evaluation)
                    evaluation_writer.add_summary(summary=evaluation_summary,
                                                  global_step=self.g_step)

            if save_during_training and (epoch % save_frequency == 0):

                save = False
                if save_best_only:
                    if (monitor == 'loss' and (not 'loss' in self.metric_names)
                            and self.training_loss < self.best_training_loss):
                        save = True
                    else:
                        i = self.metric_names.index(monitor)
                        if (monitor
                                == 'loss') and (self.metric_values[i] <
                                                self.best_metric_values[i]):
                            save = True
                        elif (monitor in ['accuracry', 'mean_iou'
                                          ]) and (self.metric_values[i] >
                                                  self.best_metric_values[i]):
                            save = True
                    if save:
                        print(
                            'Nouvelle meilleure valeur {}, sauvegarde du modèle.'
                            .format(monitor))
                    else:
                        print(
                            'Pas d\'amélioration par rapport à la meilleure valeur {} précédente, pas de sauvegarde du modèle.'
                            .format(monitor))
                else:
                    save = True

                if save:
                    self.save(model_save_dir=save_dir,
                              saver=saver,
                              tags=save_tags,
                              name=save_name,
                              include_global_step=True,
                              include_last_training_loss=True,
                              include_metrics=(len(self.metric_names) > 0))

            if self.training_loss < self.best_training_loss:
                self.best_training_loss = self.training_loss

            if epoch % eval_frequency == 0:

                for i, metric_name in enumerate(self.metric_names):
                    if (metric_name
                            == 'loss') and (self.metric_values[i] <
                                            self.best_metric_values[i]):
                        self.best_metric_values[i] = self.metric_values[i]
                    elif (metric_name in ['accuracry', 'mean_iou'
                                          ]) and (self.metric_values[i] >
                                                  self.best_metric_values[i]):
                        self.best_metric_values[i] = self.metric_values[i]

    def _evaluate(self,
                  data_generator,
                  metrics,
                  num_batches,
                  l2_regularization,
                  description='Evaluation'):

        self.sess.run(self.metrics_reset_op)

        tr = trange(num_batches, file=sys.stdout)
        tr.set_description(description)

        for step in tr:
            batch_images, batch_labels = next(data_generator)

            self.sess.run(self.metric_update_ops,
                          feed_dict={
                              self.image_input: batch_images,
                              self.labels: batch_labels,
                              self.keep_prob: 1.0,
                              self.l2_regularization_rate: l2_regularization
                          })

        self.metric_values = self.sess.run(self.metric_value_tensors)

        evaluation_results_string = ''
        for i, metric_name in enumerate(self.metric_names):
            evaluation_results_string += metric_name + \
                ': {:.4f}  '.format(self.metric_values[i])
        print(evaluation_results_string)

    def evaluate(self,
                 data_generator,
                 num_batches,
                 metrics={'loss', 'mean_iou', 'accuracy'},
                 l2_regularization=0.0,
                 dataset='val'):

        for metric in metrics:
            if not metric in ['loss', 'mean_iou', 'accuracy']:
                raise ValueError(
                    "{} n'est pas une métrique valide. Les métriques valides sont ['loss', mean_iou', 'accuracy']."
                    .format(metric))

        if not dataset in {'train', 'val'}:
            raise ValueError("`dataset` doit être soit 'train' soit 'val'..")

        self._initialize_metrics(metrics)

        self._evaluate(data_generator,
                       metrics,
                       num_batches,
                       l2_regularization,
                       description='Évaluation')

        if dataset == 'val':
            self.eval_dataset = 'val'
        else:
            self.eval_dataset = 'train'

    def predict(self, images, argmax=True):

        if argmax:
            return self.sess.run(self.predictions_argmax,
                                 feed_dict={
                                     self.image_input: images,
                                     self.keep_prob: 1.0
                                 })
        else:
            return self.sess.run(self.softmax_output,
                                 feed_dict={
                                     self.image_input: images,
                                     self.keep_prob: 1.0
                                 })

    def predict_and_save(self,
                         results_dir,
                         images_dir,
                         color_map,
                         resize=False,
                         image_file_extension='png',
                         include_unprocessed_image=False,
                         arrangement='vertical',
                         overwrite_existing=True):

        if overwrite_existing and os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)

        image_paths = glob(
            os.path.join(images_dir, '*.' + image_file_extension))
        num_images = len(image_paths)

        print('Les images segmentées seront enregistrées dans "{}".'.format(
            results_dir))

        tr = trange(num_images, file=sys.stdout)
        tr.set_description('Traitement des images')

        for i in tr:

            filepath = image_paths[i]

            image = scipy.misc.imread(filepath)
            if resize and not np.array_equal(image.shape[:2], resize):
                image = scipy.misc.imresize(image, resize)
            img_height, img_width, img_ch = image.shape

            prediction = self.predict([image], argmax=False)
            processed_image = np.asarray(print_segmentation_onto_image(
                image=image, prediction=prediction, color_map=color_map),
                dtype=np.uint8)

            if include_unprocessed_image:
                if arrangement == 'vertical':
                    output_width = img_width
                    output_height = 2 * img_height
                    processed_image = create_split_view(
                        target_size=(output_height, output_width),
                        images=[processed_image, image],
                        positions=[(0, 0), (img_height, 0)],
                        sizes=[(img_height, img_width),
                               (img_height, img_width)])
                else:
                    output_width = 2 * img_width
                    output_height = img_height
                    processed_image = create_split_view(
                        target_size=(output_height, output_width),
                        images=[processed_image, image],
                        positions=[(0, 0), (0, img_width)],
                        sizes=[(img_height, img_width),
                               (img_height, img_width)])

            scipy.misc.imsave(
                os.path.join(results_dir, os.path.basename(filepath)),
                processed_image)

    def save(self,
             model_save_dir,
             saver,
             tags=['default'],
             name=None,
             include_global_step=True,
             include_last_training_loss=True,
             include_metrics=True,
             force_save=False):

        if (not self.variables_updated) and (not force_save):
            print(
                "Abandon : Rien à sauvegarder, aucun entrainement n'a été effectuée depuis la dernière sauvegarde du modèle."
            )
            return

        if not saver in {'saved_model', 'train_saver'}:
            raise ValueError(
                "Valeur inattendue pour `saver` : Peut être soit 'saved_model' soit 'train_saver', mais a reçu '{}'."
                .format(saver))

        if self.training_loss is None:
            include_last_training_loss = False

        model_name = 'saved_model'
        if not name is None:
            model_name += '_' + name
        if include_global_step:
            self.g_step = self.sess.run(self.global_step)
            model_name += '_(globalstep-{})'.format(self.g_step)
        if include_last_training_loss:
            model_name += '_(trainloss-{:.4f})'.format(self.training_loss)
        if include_metrics:
            if self.eval_dataset == 'val':
                model_name += '_(eval_on_val_dataset)'
            else:
                model_name += '_(eval_on_train_dataset)'
            for i in range(len(self.metric_names)):
                try:
                    model_name += '_({}-{:.4f})'.format(
                        self.metric_names[i], self.metric_values[i])
                except IndexError:
                    model_name += '_{}'.format(time.time())
        if not (include_global_step or include_last_training_loss
                or include_metrics) and (name is None):
            model_name += '_{}'.format(time.time())

        if saver == 'saved_model':
            saved_model_builder = tf.saved_model.builder.SavedModelBuilder(
                os.path.join(model_save_dir, model_name))
            saved_model_builder.add_meta_graph_and_variables(sess=self.sess,
                                                             tags=tags)
            saved_model_builder.save()
        else:
            saver = tf.train.Saver(var_list=None,
                                   reshape=False,
                                   max_to_keep=5,
                                   keep_checkpoint_every_n_hours=10000.0)
            saver.save(self.sess,
                       save_path=os.path.join(model_save_dir, model_name,
                                              'variables'),
                       write_meta_graph=True,
                       write_state=True)

        self.variables_updated = False

    def load_variables(self, path):

        saver = tf.train.Saver(var_list=None)
        saver.restore(self.sess, path)

    def close(self):

        self.sess.close()
        print("La session a été clôturée.")


class UNet:

    def __init__(self,
                 n_filters=4,
                 dilation_rate=1,
                 n_classes=8,
                 activation='relu'):

        self.n_filters = n_filters
        self.dilation_rate = dilation_rate
        self.n_classes = n_classes
        self.activation = activation

    def Unet(self, n_filters, dilation_rate, n_classes, activation):
        # Définir la forme du lot d'entrée
        inputs = Input(shape=(256, 256, 3))

        conv1 = Conv2D(n_filters * 1, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(inputs)
        conv1 = Conv2D(n_filters * 1, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2),
                             data_format='channels_last')(conv1)

        conv2 = Conv2D(n_filters * 2, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(pool1)
        conv2 = Conv2D(n_filters * 2, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2),
                             data_format='channels_last')(conv2)

        conv3 = Conv2D(n_filters * 4, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(pool2)
        conv3 = Conv2D(n_filters * 4, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2),
                             data_format='channels_last')(conv3)

        conv4 = Conv2D(n_filters * 8, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(pool3)
        conv4 = Conv2D(n_filters * 8, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2),
                             data_format='channels_last')(conv4)

        conv5 = Conv2D(n_filters * 16, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(pool4)
        conv5 = Conv2D(n_filters * 16, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(conv5)
        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)

        conv6 = Conv2D(n_filters * 8, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(up6)
        conv6 = Conv2D(n_filters * 8, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(conv6)
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)

        conv7 = Conv2D(n_filters * 4, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(up7)
        conv7 = Conv2D(n_filters * 4, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(conv7)
        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)

        conv8 = Conv2D(n_filters * 2, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(up8)
        conv8 = Conv2D(n_filters * 2, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(conv8)
        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)

        conv9 = Conv2D(n_filters * 1, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(up9)
        conv9 = Conv2D(n_filters * 1, (3, 3),
                       activation=activation,
                       padding='same',
                       dilation_rate=dilation_rate)(conv9)
        conv10 = Conv2D(n_classes, (1, 1),
                        activation='softmax',
                        padding='same',
                        dilation_rate=dilation_rate)(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        return model


class SegNet:

    def __init__(self, pretrained_weights=None, input_size=(256, 256, 3)):

        self.pretrained_weights = pretrained_weights
        self.input_size = input_size

    def segnet(pretrained_weights, input_size):
        inputs = Input(input_size)
        # étape 1
        conv1 = Conv2D(64,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(inputs)
        conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(64,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # étape 2
        conv2 = Conv2D(128,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(128,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # étape 3
        conv3 = Conv2D(256,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(256,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(256,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # étape 4
        conv4 = Conv2D(512,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(512,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(512,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # étape 5
        conv5 = Conv2D(1024,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(pool4)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(1024,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(1024,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization()(conv5)
        pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

        # étape 6
        up6 = Conv2D(1024,
                     2,
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal')(
                         UpSampling2D(size=(2, 2))(pool5))
        merge6 = concatenate([conv5, up6], axis=3)
        conv6 = Conv2D(1024,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(merge6)
        conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(1024,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv6)
        conv6 = BatchNormalization()(conv6)

        # étape 7
        up7 = Conv2D(512,
                     2,
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal')(
                         UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv4, up7], axis=3)
        conv7 = Conv2D(512,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(merge7)
        conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(512,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv7)
        conv7 = BatchNormalization()(conv7)

        # étape 8
        up8 = Conv2D(256,
                     2,
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal')(
                         UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv3, up8], axis=3)
        conv8 = Conv2D(256,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(merge8)
        conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(256,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv8)
        conv8 = BatchNormalization()(conv8)

        # étape 9
        up9 = Conv2D(128,
                     2,
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal')(
                         UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv2, up9], axis=3)
        conv9 = Conv2D(128,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(merge9)
        conv9 = BatchNormalization()(conv9)
        conv9 = Conv2D(128,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization()(conv9)

        # étape 10
        up10 = Conv2D(64,
                      3,
                      activation='relu',
                      padding='same',
                      kernel_initializer='he_normal')(
                          UpSampling2D(size=(2, 2))(conv9))
        merge10 = concatenate([conv1, up10], axis=3)
        conv10 = Conv2D(64,
                        3,
                        activation='relu',
                        padding='same',
                        kernel_initializer='he_normal')(merge10)
        conv10 = BatchNormalization()(conv10)
        conv10 = Conv2D(8, 1, activation='softmax')(conv10)

        model = Model(inputs, conv10)

        if (pretrained_weights):
            model.load_weights(pretrained_weights)

        return model