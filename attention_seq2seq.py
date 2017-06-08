from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import pdb
import contextlib
import math


# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from load_embeddings import load_embedding
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope

from tensorflow.python.util import nest



def embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                cell,
                                encoder_cell,
                                session,
                                vocabSize,
                                pathToEmbeddings,
                                vocab,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1,
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
  """Embedding sequence-to-sequence model with attention.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. It keeps the outputs of this
  RNN at every step to use for attention later. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  input_size]). Then it runs attention decoder, initialized with the last
  encoder state, on embedded decoder_inputs and attending to encoder outputs.

  Warning: when output_projection is None, the size of the attention vectors
  and variables will be made proportional to num_decoder_symbols, can be large.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: core_rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial RNN state (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_seq2seq".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x num_decoder_symbols] containing the generated
        outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with variable_scope.variable_scope(
      scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
    dtype = scope.dtype
    # Encoder.
    #encoder_cell = copy.deepcopy(cell)
    encoder_cell = PretrainedEmbeddingWrapper(
        encoder_cell,
        session, 
        vocab,
        vocabSize,
        pathToEmbeddings,
        embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    encoder_outputs, encoder_state = core_rnn.static_rnn(
        encoder_cell, encoder_inputs, dtype=dtype)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [
        array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs
    ]
    attention_states = array_ops.concat(top_states, 1)

    # Decoder.
    output_size = None
    if output_projection is None:
      cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
      output_size = num_decoder_symbols

    if isinstance(feed_previous, bool):
      return seq2seq.embedding_attention_decoder(
          decoder_inputs,
          encoder_state,
          attention_states,
          cell,
          num_decoder_symbols,
          embedding_size,
          num_heads=num_heads,
          output_size=output_size,
          output_projection=output_projection,
          feed_previous=feed_previous,
          initial_state_attention=initial_state_attention)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=reuse) as scope:
        outputs, state = seq2seq.embedding_attention_decoder(
            decoder_inputs,
            encoder_state,
            attention_states,
            cell,
            num_decoder_symbols,
            embedding_size,
            num_heads=num_heads,
            output_size=output_size,
            output_projection=output_projection,
            feed_previous=feed_previous_bool,
            update_embedding_for_previous=False,
            initial_state_attention=initial_state_attention)
        state_list = [state]
        if nest.is_sequence(state):
          state_list = nest.flatten(state)
        return outputs + state_list

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
    state_list = outputs_and_state[outputs_len:]
    state = state_list[0]
    if nest.is_sequence(encoder_state):
      state = nest.pack_sequence_as(
          structure=encoder_state, flat_sequence=state_list)
    return outputs_and_state[:outputs_len], state


########################################################################
def pretrained_embedding_rnn_seq2seq(encoder_inputs,
                                     decoder_inputs,
                                     cell,
                                     encoder_cell,
                                     session,
                                     vocabSize,
                                     pathToEmbeddings,
                                     vocab,
                                     num_encoder_symbols,
                                     num_decoder_symbols,
                                     embedding_size,
                                     output_projection=None,
                                     feed_previous=False,
                                     dtype=None,
                                     scope=None):
    """Embedding RNN sequence-to-sequence model.

    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_encoder_symbols x input_size]). Then it runs an RNN to encode
    embedded encoder_inputs into a state vector. Next, it embeds decoder_inputs
    by another newly created embedding (of shape [num_decoder_symbols x
    input_size]). Then it runs RNN decoder, initialized with the last
    encoder state, on embedded decoder_inputs.

    Args:
      encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      cell: core_rnn_cell.RNNCell defining the cell function and size.
      num_encoder_symbols: Integer; number of symbols on the encoder side.
      num_decoder_symbols: Integer; number of symbols on the decoder side.
      embedding_size: Integer, the length of the embedding vector for each symbol.
      output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_decoder_symbols] and B has
        shape [num_decoder_symbols]; if provided and feed_previous=True, each
        fed previous output will first be multiplied by W and added B.
      feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
        of decoder_inputs will be used (the "GO" symbol), and all other decoder
        inputs will be taken from previous outputs (as in embedding_rnn_decoder).
        If False, decoder_inputs are used as given (the standard decoder case).
      dtype: The dtype of the initial state for both the encoder and encoder
        rnn cells (default: tf.float32).
      scope: VariableScope for the created subgraph; defaults to
        "embedding_rnn_seq2seq"

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors. The
          output is of shape [batch_size x cell.output_size] when
          output_projection is not None (and represents the dense representation
          of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
          when output_projection is None.
        state: The state of each decoder cell in each time-step. This is a list
          with length len(decoder_inputs) -- one item for each time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].

    """
    pdb.set_trace()
    with variable_scope.variable_scope(scope or "embedding_rnn_seq2seq", reuse=None) as scope:
        if dtype is not None:
            scope.set_dtype(dtype)
        else:
            dtype = scope.dtype

        # Encoder.
        
		#encoder_cell = copy.deepcopy(cell)

        # pdb.set_trace()
        encoder_cell = PretrainedEmbeddingWrapper(encoder_cell, session, vocab, vocabSize,
                                                                pathToEmbeddings, embedding_classes=num_encoder_symbols,
                                                                embedding_size=embedding_size)
        _, encoder_state = core_rnn.static_rnn(
            encoder_cell, encoder_inputs, dtype=dtype)

        # Decoder.
        if output_projection is None:
            cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)

        if isinstance(feed_previous, bool):
            return seq2seq.embedding_rnn_decoder(
                decoder_inputs,
                encoder_state,
                cell,
                num_decoder_symbols,
                embedding_size,
                output_projection=output_projection,
                feed_previous=feed_previous)

        # If feed_previous is a Tensor, we construct 2 graphs and use cond.
        def decoder(feed_previous_bool):
            reuse = None if feed_previous_bool else True
            with variable_scope.variable_scope(
                    variable_scope.get_variable_scope(), reuse=reuse) as scope:
                outputs, state = seq2seq.embedding_rnn_decoder(
                    decoder_inputs,
                    encoder_state,
                    cell,
                    num_decoder_symbols,
                    embedding_size,
                    output_projection=output_projection,
                    feed_previous=feed_previous_bool,
                    update_embedding_for_previous=False)
                state_list = [state]
                if nest.is_sequence(state):
                    state_list = nest.flatten(state)
                return outputs + state_list

        outputs_and_state = control_flow_ops.cond(feed_previous,
                                                  lambda: decoder(True),
                                                  lambda: decoder(False))
        outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
        state_list = outputs_and_state[outputs_len:]
        state = state_list[0]
        if nest.is_sequence(encoder_state):
            state = nest.pack_sequence_as(
                structure=encoder_state, flat_sequence=state_list)
        return outputs_and_state[:outputs_len], state



class PretrainedEmbeddingWrapper(RNNCell):
  """Operator adding input embedding to the given cell.

  Note: in many cases it may be more efficient to not use this wrapper,
  but instead concatenate the whole sequence of your inputs in time,
  do the embedding on this batch-concatenated sequence, then split it and
  feed into your RNN.
  """

  def __init__(self, cell, sess, vocab, vocabSize, pathToEmbedding, embedding_classes, embedding_size, initializer=None,
               reuse=None):
    """Create a cell with an added input embedding.

    Args:
      cell: an RNNCell, an embedding will be put before its inputs.
      embedding_classes: integer, how many symbols will be embedded.
      embedding_size: integer, the size of the vectors we embed into.
      initializer: an initializer to use when creating the embedding;
        if None, the initializer from variable scope or a default one is used.
      reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.

    Raises:
      TypeError: if cell is not an RNNCell.
      ValueError: if embedding_classes is not positive.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not RNNCell.")
    if embedding_classes <= 0 or embedding_size <= 0:
      raise ValueError("Both embedding_classes and embedding_size must be > 0: "
                       "%d, %d." % (embedding_classes, embedding_size))
    self._cell = cell
    self._embedding_classes = embedding_classes
    self._embedding_size = embedding_size
    self._initializer = initializer
    self._reuse = reuse
    self._sess = sess
    self._vocab = vocab
    self._vocabSize = vocabSize
    self._pathToEmbedding = pathToEmbedding
    self._looked_embeddings = False

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def __call__(self, inputs, state, scope=None):
    """Run the cell on embedded inputs."""
    with _checked_scope(self, scope or "embedding_wrapper", reuse=self._reuse):
      with ops.device("/cpu:0"):
        if self._initializer:
          initializer = self._initializer
        elif vs.get_variable_scope().initializer:
          initializer = vs.get_variable_scope().initializer
        else:
          # Default initializer for embeddings should have variance=1.
          sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
          initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)

        if type(state) is tuple:
          data_type = state[0].dtype
        else:
          data_type = state.dtype

        embedding = vs.get_variable(
            "embedding", [self._embedding_classes, self._embedding_size],
            initializer=initializer,
            dtype=data_type)

        if (not self._looked_embeddings):
          load_embedding(self._sess, self._vocab, embedding, self._pathToEmbedding, self._embedding_size, self._vocabSize)
          self._looked_embeddings = True
        embedded = embedding_ops.embedding_lookup(
            embedding, array_ops.reshape(inputs, [-1]))
    return self._cell(embedded, state)


  #################################################################################################

@contextlib.contextmanager
def _checked_scope(cell, scope, reuse=None, **kwargs):
  if reuse is not None:
    kwargs["reuse"] = reuse
  with vs.variable_scope(scope, **kwargs) as checking_scope:
    scope_name = checking_scope.name
    if hasattr(cell, "_scope"):
      cell_scope = cell._scope  # pylint: disable=protected-access
      if cell_scope.name != checking_scope.name:
        raise ValueError(
            "Attempt to reuse RNNCell %s with a different variable scope than "
            "its first use.  First use of cell was with scope '%s', this "
            "attempt is with scope '%s'.  Please create a new instance of the "
            "cell if you would like it to use a different set of weights.  "
            "If before you were using: MultiRNNCell([%s(...)] * num_layers), "
            "change to: MultiRNNCell([%s(...) for _ in range(num_layers)]).  "
            "If before you were using the same cell instance as both the "
            "forward and reverse cell of a bidirectional RNN, simply create "
            "two instances (one for forward, one for reverse).  "
            "In May 2017, we will start transitioning this cell's behavior "
            "to use existing stored weights, if any, when it is called "
            "with scope=None (which can lead to silent model degradation, so "
            "this error will remain until then.)"
            % (cell, cell_scope.name, scope_name, type(cell).__name__,
               type(cell).__name__))
    else:
      weights_found = False
      try:
        with vs.variable_scope(checking_scope, reuse=True):
          vs.get_variable("weights")
        weights_found = True
      except ValueError:
        pass
      if weights_found and reuse is None:
        raise ValueError(
            "Attempt to have a second RNNCell use the weights of a variable "
            "scope that already has weights: '%s'; and the cell was not "
            "constructed as %s(..., reuse=True).  "
            "To share the weights of an RNNCell, simply "
            "reuse it in your second calculation, or create a new one with "
            "the argument reuse=True." % (scope_name, type(cell).__name__))

    # Everything is OK.  Update the cell's scope and yield it.
    cell._scope = checking_scope  # pylint: disable=protected-access
    yield checking_scope

