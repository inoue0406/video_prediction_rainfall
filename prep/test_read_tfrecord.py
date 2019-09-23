import numpy as np
import tensorflow as tf

def parser(self, serialized_example):
    """
    Parses a single tf.train.Example into images, states, actions, etc tensors.
    """
    features = dict()
    for i in range(self._max_sequence_length):
        for example_name, (name, shape) in self.state_like_names_and_shapes.items():
            if example_name == 'images':  # special handling for image
                features[name % i] = tf.FixedLenFeature([1], tf.string)
            else:
                features[name % i] = tf.FixedLenFeature(shape, tf.float32)
    for i in range(self._max_sequence_length - 1):
        for example_name, (name, shape) in self.action_like_names_and_shapes.items():
            features[name % i] = tf.FixedLenFeature(shape, tf.float32)

    # check that the features are in the tfrecord
    for name in features.keys():
        if name not in self._dict_message['features']['feature']:
            raise ValueError('Feature with name %s not found in tfrecord. Possible feature names are:\n%s' %
                             (name, '\n'.join(sorted(self._dict_message['features']['feature'].keys()))))

    # parse all the features of all time steps together
    features = tf.parse_single_example(serialized_example, features=features)

    state_like_seqs = OrderedDict([(example_name, []) for example_name in self.state_like_names_and_shapes])
    action_like_seqs = OrderedDict([(example_name, []) for example_name in self.action_like_names_and_shapes])
    for i in range(self._max_sequence_length):
        for example_name, (name, shape) in self.state_like_names_and_shapes.items():
            state_like_seqs[example_name].append(features[name % i])
    for i in range(self._max_sequence_length - 1):
        for example_name, (name, shape) in self.action_like_names_and_shapes.items():
            action_like_seqs[example_name].append(features[name % i])

    # for this class, it's much faster to decode and preprocess the entire sequence before sampling a slice
    _, image_shape = self.state_like_names_and_shapes['images']
    state_like_seqs['images'] = self.decode_and_preprocess_images(state_like_seqs['images'], image_shape)

    state_like_seqs, action_like_seqs = \
             self.slice_sequences(state_like_seqs, action_like_seqs, self._max_sequence_length)
    return state_like_seqs, action_like_seqs

def filter(self, serialized_example):
    return tf.convert_to_tensor(True)

if __name__ == '__main__':
    """
    Test for tf.dataset parsing
    """
    
    filename = "../data/bair/train/traj_10174_to_10429.tfrecords"
    
    filenames = [filename]
    
    dataset = tf.data.TFRecordDataset(filenames,buffer_size=8*1024*1024)
    
    dataset = dataset.filter(filter)
    
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=1024, count=self.num_epochs))
    def _parser(serialized_example):
        state_like_seqs, action_like_seqs = parser(serialized_example)
        seqs = OrderedDict(list(state_like_seqs.items()) + list(action_like_seqs.items()))
        return seqs

    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        _parser, batch_size, drop_remainder=True, num_parallel_calls=num_parallel_calls))
    dataset = dataset.prefetch(batch_size)
    
#itr = raw_dataset.make_one_shot_iterator()
#    x = itr.get_next()
#
#    sess = tf.Session()
#    images = sess.run(x)
#
#    image = tf.decode_raw(image_buffer, tf.uint8)

    import pdb;pdb.set_trace()
    
#    for raw_record in raw_dataset.take(10):
#        print(repr(raw_record))

