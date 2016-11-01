# from ipywidgets import interact
import tensorflow as tf
session = tf.InteractiveSession()
print "here?"
from data import decode_output_sequences
from model2 import Seq2SeqProgramModel

from program_generator import ProgramGenerator, SYMBOLS, SYMBOL_TO_IDX, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN 

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large, zoneout.")
flags.DEFINE_string("data_path", None, "data_path")
FLAGS = flags.FLAGS

from program_generator import ProgramGenerator, SYMBOLS, SYMBOL_TO_IDX, INPUT_SEQ_LEN, OUTPUT_SEQ_LEN 

program_generator = ProgramGenerator(batch_size=10, program_length=1, num_len=2)
x, y = program_generator.next_batch()

input_strings = decode_output_sequences(x, symbols=SYMBOLS)
target_strings = decode_output_sequences(y, symbols=SYMBOLS)

print(" Inputs:", input_strings)
print("Targets:", target_strings)
session.close()
tf.reset_default_graph()
session = tf.InteractiveSession()

hidden_units = 320
num_layers = 2
training_batch_size = 128
num_symbols = len(SYMBOL_TO_IDX)

program_model = Seq2SeqProgramModel(session=session,
                                hidden_units=hidden_units, 
                                num_layers=num_layers,
                                input_sequence_len = INPUT_SEQ_LEN,
                                output_sequence_len = OUTPUT_SEQ_LEN,
                                num_input_symbols = num_symbols,
                                num_output_symbols = num_symbols,
                                batch_size=training_batch_size,
                                symbols=SYMBOLS,
                                scope='model')

program_model.init_variables()

program_generator = ProgramGenerator(batch_size=training_batch_size, program_length=1, num_len=2)

print("Finished building model")
program_model.fit(program_generator, 
                  num_epochs=20000, 
                  batches_per_epoch=128)
print("Finished training")


## Restore previously trained model with 320 hidden units took about 10h to train on an AWS instance.
#saver = tf.train.Saver()
#saver.restore(session, "trained_model/model_(2, 3).ckpt")



## View predictions
#from random import seed

#seed(101)
#test_generator = ProgramGenerator(batch_size=training_batch_size, num_len=2, program_length=3)

#x, y = test_generator.next_batch(validation=True)

#input_strings = decode_output_sequences(x, symbols=SYMBOLS)
#target_strings = decode_output_sequences(y, symbols=SYMBOLS)

#model_output = program_model.predict(x)

#pred_strings = decode_output_sequences(model_output, symbols=SYMBOLS)

#def view_prediction(i):
    #print(input_strings[i][::-1].strip('_'))
    #print("--------")
    #print("Targ:", target_strings[i].strip('_'))
    #print("Pred:", pred_strings[i].strip('_'))
#print 'a = interact(view_prediction, i=(0, training_batch_size - 1))'
