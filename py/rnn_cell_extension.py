
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip     # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.python import shape
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope


class OutputProjectionWrapperWithParameter(rnn_cell.RNNCell):
    '''
    Similar to OutputProjectionWrapper, but with user provided W and b. 
    '''

    def __init__(self, cell, output_size, w, b, projected_context = None, embedding_scope = None):
        """Create a cell with output projection.
        Args:
          cell: an RNNCell, a projection with w and b is added to it.
          output_size: integer, the size of the output after projection.
          w: the projection matrix, with shape [cell.output_size, cell.output_size]
          b: the bias, with shape [output_size]
        Raises:
          TypeError: if cell is not an RNNCell.
          ValueError: if output_size is not positive.
        """
        if not isinstance(cell, rnn_cell.RNNCell):
          raise TypeError("The parameter cell is not RNNCell.")
        if output_size < 1:
          raise ValueError("Parameter output_size must be > 0: %d." % output_size)
        self._cell = cell
        self._output_size = output_size
        self._w = w
        self._b = b
        self._projected_context = projected_context
        if projected_context == None:
            self.with_context = False
        else:
            self.with_context = True
        self.embedding_scope = embedding_scope
        
    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._output_size
    
    def __call__(self, inputs, state, scope=None):
        


        """Run the cell and output projection on inputs, starting from state."""
        output, res_state = self._cell(inputs, state)
        # Default scope: "OutputProjectionWrapper"
        with variable_scope.variable_scope(self.embedding_scope):
            try:
                embedding_wo = tf.get_variable("embedding_wo",[self._cell.state_size,self._output_size])
                if self.with_context:                
                    embedding_wc = tf.get_variable("embedding_wc",[self._cell.state_size,self._output_size])
            except:
                variable_scope.get_variable_scope().reuse_variables()
                embedding_wo = tf.get_variable("embedding_wo",[self._cell.state_size,self._output_size])
                if self.with_context:                
                    embedding_wc = tf.get_variable("embedding_wc",[self._cell.state_size,self._output_size])

            
            


        with variable_scope.variable_scope(scope or type(self).__name__):
            outputr = array_ops.reshape(output, [-1,1, self._cell.state_size])
            projected = array_ops.reshape(tf.batch_matmul(outputr, self._w),[-1,self._cell.state_size])
            
            if self.with_context:
                projected = tf.matmul(projected, embedding_wo) + tf.matmul(self._projected_context, embedding_wc) + self._b
            else:
                projected = tf.matmul(projected, embedding_wo) + self._b   

        return projected, res_state
