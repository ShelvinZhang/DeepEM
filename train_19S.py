import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os,sys,shutil,time
from vgg19 import Vgg19
from utils import load_train
from args_vgg19_19S import Train_Args

def train():
    args = Train_Args() ## load args parameters to a class named args
    train_start = time.time()
    time_start = time.time()

    checkpoint_dir = args.model_save_path
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
        
    output_name = args.model_save_path + "training_messages.txt"
    output = open(output_name, 'w')
    print("read data start.",flush=True)
    output.write("read data start.\n")

    train_x, train_y, test_x, test_y = load_train(args)
    print("shape of train_x: " , train_x.shape)
    time_end = time.time()
    print("\nread done! totally cost: %.5f \n" %(time_end - time_start),flush=True)
    output.write("read done! totally cost: " + str(time_end - time_start) +'\n')
    output.flush()
    print("training start.",flush=True)
    # copy argument file
    srcfile = args.model_filename
    dstfile = args.model_save_path + srcfile
    shutil.copyfile(srcfile,dstfile)
    shutil.copyfile(args.args_filename,args.model_save_path + args.args_filename)

    time_start = time.time()

    tot_cost = []
    plot = []
    plot_train = []
    best_test_accuracy = 0
    best_train_accuracy = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # training start
    with tf.Session() as sess:
        deepem = Vgg19(args)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 100)
        sess.run(tf.global_variables_initializer())
        print("train size is %d " % len(train_x), flush=True)

        for e in range(args.num_epochs):
            print('\n=============== Epoch %d/%d ==============='% (e + 1,args.num_epochs),flush=True)
            output.write("\n=============== Epoch " + str(e + 1) + "/" + str(args.num_epochs) + " ===============\n")
            cost = []
            # num_batch = len(train_x) // args.batch_size
            num_batch = int(np.ceil(len(train_x)/args.batch_size))
            print("num_batch is %d" % num_batch,flush=True)
            test_train_pred = []
            test_train_y = []
            for i in range(num_batch):
                batch_x = train_x[args.batch_size*i:args.batch_size*(i+1)]
                batch_y = train_y[args.batch_size*i:args.batch_size*(i+1)]
                # batch_x = batch_x.reshape((args.batch_size, args.resize, args.resize, 1))
                loss,pred_train,lr,_= sess.run([deepem.loss, deepem.pred,deepem.lr, deepem.optimizer], {deepem.X:batch_x, deepem.Y: batch_y})
                test_train_pred[args.batch_size*i:args.batch_size*(i+1)] = pred_train[:]
                test_train_y[args.batch_size*i:args.batch_size*(i+1)] = batch_y[:]
                cost.append(loss)
                if i % 10 == 0:
                    print('lr: %.8f loss: %.6f' % (lr, np.mean(cost)),flush=True)

            tot_cost.append([np.mean(cost)])
            output.write("average loss: " + str(np.mean(cost)) + '\n')
            output.flush()
            # print("test_train_pred",test_train_pred)
            test_train_pred = np.asarray(test_train_pred)
            print("avg = %.6f , min = %.6f, max = %.6f "% (test_train_pred.mean(),test_train_pred.min(),test_train_pred.max()))
            threhold = 0.5
            test_train_pred[test_train_pred<=threhold] = 0
            test_train_pred[test_train_pred>threhold] = 1
            accuracy_train = np.sum(np.equal(test_train_pred,test_train_y))/len(test_train_y)

            if accuracy_train > best_train_accuracy:
                best_train_accuracy = accuracy_train

            plot_train.append(1- accuracy_train)
            print("train accuracy: %.6f" % accuracy_train,flush=True)
            print("best_train_accuracy: %.6f" % best_train_accuracy,flush=True)
            output.write("train accuracy: " + str(accuracy_train) + '\n')
            output.flush()

            # start testing
            if (e+1) % 5 == 0 or (e+1) == args.num_epochs:
                print("\ntesting start.",flush=True)
                # num_batch = len(test_x) // args.batch_size
                num_batch = int(np.ceil(len(test_x)/args.batch_size))
                print("num_batch is %d" % num_batch,flush=True)
                test_pred = []
                for i in range(num_batch):
                    batch_x = test_x[args.batch_size*i:args.batch_size*(i+1)]
                    # batch_x = batch_x.reshape((args.batch_size, args.resize, args.resize, 1))
                    batch_y = test_y[args.batch_size*i:args.batch_size*(i+1)]
                    pred = sess.run(deepem.pred,feed_dict={deepem.X: batch_x})
                    test_pred[args.batch_size*i:args.batch_size*(i+1)] = pred[:]
                test_pred = np.asarray(test_pred)
                print("avg = %.6f , min = %.6f, max = %.6f "% (test_pred.mean(),test_pred.min(),test_pred.max()))
                threhold = 0.5
                test_pred[test_pred<=threhold] = 0
                test_pred[test_pred>threhold] = 1
                accuracy = np.sum(np.equal(test_pred,test_y))/len(test_y)
                plot.append(1- accuracy)
                print("testing set accuracy: %.6f" % accuracy,flush=True)
                output.write("testing set accuracy: " + str(accuracy) + '\n')
                output.write("best_test_accuracy: " + str(best_test_accuracy) + '\n')
                output.flush()

                ckpt_path = os.path.join(checkpoint_dir, 'model.ckpt')
                saver.save(sess, ckpt_path, global_step = e+1)
                print("model saved!")
                
                if accuracy > best_test_accuracy:
                    best_test_accuracy = accuracy
                
                print("best_test_accuracy: %.6f" % best_test_accuracy,flush=True)
        time_end = time.time()
        print("\ntraining done! totally cost: %.5f \n" %(time_end - time_start),flush=True)
        output.write("training done! totally cost: " + str(time_end - time_start) + '\n')
        output.flush()

    train_end = time.time()
    print("\ntrain done! totally cost: %.5f \n" %(train_end - train_start),flush=True)
    print("best_train_accuracy: %.6f " % best_train_accuracy)
    print("best_test_accuracy: %.6f " % best_test_accuracy)
    output.write("best_train_accuracy: " + str(best_train_accuracy) + '\n')
    output.write("best_test_accuracy: " + str(best_test_accuracy) + '\n')
    output.flush()
    output.close

    # draw picture of loss and error
    plt.subplot(211)
    plt.title('error r:test  g:train')
    plt.xlabel("epoch")
    plt.ylabel("error")
    plt.grid(True)
    plt.plot(range(0,5*len(plot),5),plot,"r^--",range(len(plot_train)),plot_train,"g^--")
    
    plt.subplot(212)
    plt.title('loss of training')
    plt.plot(range(len(tot_cost)),tot_cost,"b^--")
    plt.grid(True)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(args.model_save_path + "loss_error.png")
    plt.show()

if __name__ == '__main__':
    train()

