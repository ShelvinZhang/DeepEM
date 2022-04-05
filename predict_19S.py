import tensorflow as tf
import numpy as np
import mrcfile
import os,sys,time
from vgg19 import Vgg19
from utils import mapstd,sub_img,non_max_suppression,load_predict
from args_vgg19_19S import Predict_Args

def predict():
    args = Predict_Args()
    deepem = Vgg19(args)
    checkpoint_dir = args.model_save_path
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('Restore model failed!',flush=True)
        if not os.path.exists(args.result_path): 
            os.mkdir(args.result_path)     

        print("start predicting....\n",flush=True)
        time_start = time.time()

        for num in range(args.start_mic_num, args.end_mic_num + 1):
            mrc_name = args.data_path + args.name_prefix + str(num).zfill(args.name_length) + args.name_suffix + ".mrc"
            if not os.path.exists(mrc_name):
                print("%s is not exist!" % mrc_name)
                continue

            print("\nprocessing mrc %s..." % mrc_name,flush=True)
            output_name = args.result_path + args.name_prefix + str(num).zfill(args.name_length)  + args.name_suffix + '.box'
            output = open(output_name, 'w')
            test_x,test_index = load_predict(args,mrc_name)
            test_x = np.asarray(test_x).reshape(len(test_x),args.boxsize,args.boxsize,1)

            num_batch = len(test_x) // args.batch_size
            print("num_of_box is %d" % len(test_x),flush=True)
            particle = []
            scores = []
            for i in range(num_batch):
                batch_x = test_x[args.batch_size*i:args.batch_size*(i+1)]
                batch_test_index = test_index[args.batch_size*i:args.batch_size*(i+1)]
                pred = sess.run(deepem.pred,feed_dict={deepem.X: batch_x})
                # print("pred: avg = %.10f, max = %.10f " % ( pred.mean(), pred.max())) 
                for i in range(len(batch_test_index)):
                    if pred[i] > args.accuracy:
                        #print("%d.mrc %d %d %.10f"%(num,batch_test_index[i][0],batch_test_index[i][1],pred[i]),flush=True)
                        particle.append([batch_test_index[i][0], batch_test_index[i][1]])
                        scores.append(pred[i])
            print("%d particles detected in %s!" % (len(particle),mrc_name))
            if len(particle) == 0:
                output.close
                continue
            particle = np.asarray(particle)
            scores = np.asarray(scores)
            # remove overlapping particles
            result = non_max_suppression(particle, scores, args.boxsize,args.threhold )
            for i in range(len(result)):
                #print("%d.mrc %d %d "%(num,result[i][0],result[i][1]),flush=True)
                output.write(str(result[i][0])+'\t'+ str(result[i][1])+'\t'+str(args.boxsize)+'\t'+str(args.boxsize) +'\n')
                output.flush()
            print("%d particles left in %s!" % (len(result),mrc_name))
            output.close

    time_end = time.time()
    print("\ntotal %d mrc pictures." % (args.end_mic_num - args.start_mic_num + 1))
    print("predicting done! totally cost: %.5f \n" %(time_end - time_start),flush=True)
    print("cost of every single mrc file: %.5f \n" %((time_end - time_start)/(args.end_mic_num - args.start_mic_num + 1)),flush=True)

if __name__ == '__main__':
    predict()

