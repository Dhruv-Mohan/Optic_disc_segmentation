
_SUMMARY_           =   True
_BATCH_SIZE_        =   1
_IMAGE_WIDTH_      =   256*4
_IMAGE_HEIGHT_     =   256*4
_IMAGE_CSPACE_    =    3
_CLASSES_             =   1
_MODEL_NAME_       =  'FRRN_C'
_ITERATIONS_         =   100000
_LEARNING_RATE_   =   0.02#0.05 0.08
_SAVE_DIR_            =   'G:/TFmodels/fold1c/'
_SAVE_INTERVAL_    =  2500
_RESTORE_              =  True
_TEST_                   =  True
_PREDICTION_PATH   = 'G:/Pri/'
_DATASET_ = 'G:/fold3.tfrecords'
_TEST_PATH_ = 'G:/Datasets/MD/Test1/M/'
_GT_SUFFIX_ =  '_ODsegSoftmap.png'
index = 0
Test_GT = 'G:/Datasets/MD/Test1/mask/'
gt_masks = os.listdir(Test_GT)
Dice_coeffs = []
Jacard_coeff = []


def get_dice_loss(Mat1, Mat2):
    if len(Mat1.shape) is 3:
        Mat1= cv2.cvtColor(Mat1, cv2.COLOR_BGR2GRAY)
    if len(Mat2.shape) is 3:
        Mat2= cv2.cvtColor(Mat2, cv2.COLOR_BGR2GRAY)
    Mat1 = np.float32(Mat1)
    Mat2 = np.float32(Mat2)
    Mat1 = Mat1/(np.max(Mat1) +1e-10)
    Mat2 = Mat2/(np.max(Mat2) + 1e-10)
    Mat1 = np.reshape(Mat1, (-1, _IMAGE_WIDTH_ * _IMAGE_HEIGHT_))
    Mat2 = np.reshape(Mat2, (-1,_IMAGE_WIDTH_ * _IMAGE_HEIGHT_))
    Intersection = np.sum(Mat1 * Mat2) 
    Union = np.sum(Mat1 + Mat2) +1e-10
    Dice_Coeff = 2*Intersection / Union
    Jacard_coeff = Intersection/ (Union - Intersection)
    return Dice_Coeff, Jacard_coeff


def writer_pre_proc_seg_test(images):
    print('Adding image Preproc')
    resized_images = tf.image.resize_images(images, size=[_IMAGE_WIDTH_,_IMAGE_HEIGHT_])
    resized_images=tf.image.per_image_standardization(resized_images)
    reshaped_images = tf.reshape(resized_images, [-1,_IMAGE_HEIGHT_*_IMAGE_WIDTH_*_IMAGE_CSPACE_])
    return reshaped_images 

def construct_segmap(image, GT_name=None):
    global index
    global Test_GT
    global gt_masks
    global Dice_coeffs
    global Jacard_coeff
    #image = cv2.resize(image, (1024,1024), interpolation=cv2.INTER_LANCZOS4)

    GT = cv2.imread(Test_GT+GT_name)
    GT = cv2.resize(GT, ( _IMAGE_WIDTH_ , _IMAGE_HEIGHT_), interpolation=cv2.INTER_LANCZOS4)
    #GT = cv2.imread('G:/Datasets/MD/Test1/2i.png')
    zeros = np.zeros(GT.shape)
    
    ret,thresh = cv2.threshold(image,0.5,255,cv2.THRESH_BINARY)
    thresh = np.uint8(thresh)
    zeros = np.uint8(zeros)
    _, c,h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_c = max(c, key=cv2.contourArea)
    cv2.drawContours(zeros, [max_c], 0, (255,255,255), -1)
    thresh = cv2.cvtColor(zeros, cv2.COLOR_BGR2GRAY)
    thresh = np.float32(thresh)
    cv2.imshow('thresh', thresh)
    cv2.imshow('GT', GT)
    cv2.waitKey(1)
    DC, JC = get_dice_loss(thresh, GT)
    Dice_coeffs.extend([DC])
    Jacard_coeff.extend([JC])
    print("DICE COEFF:",Dice_coeffs[index] )
    print("JACARD COEFF:",Jacard_coeff[index] )
    cv2.imwrite(GT_name, thresh)
    index = index +1

    return thresh


def main():
    '''
    Main fxn used for training and testing
    '''
    global Dice_coeffs
    global Jacard_coeff

    #Segmentation reader init
    with tf.device('/cpu:0'): 
        dummy_reader = Dataset_reader_segmentation(_DATASET_)
        dummy_reader.pre_process_image(writer_pre_proc)
        dummy_reader.pre_process_mask(writer_pre_proc_mask)
        dummy_reader.pre_process_weights(writer_pre_proc_weight)

    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    #Construct model
    with tf.name_scope(_MODEL_NAME_):
        Simple_DNN = Model(Model_name=_MODEL_NAME_, Summary=_SUMMARY_, \
            Batch_size=_BATCH_SIZE_, Image_width=_IMAGE_WIDTH_, Image_height=_IMAGE_HEIGHT_, Image_cspace=_IMAGE_CSPACE_, Classes=_CLASSES_, Save_dir=_SAVE_DIR_)

        Simple_DNN.Construct_Model()

        #Set loss
        Simple_DNN.Set_loss()

    #Set optimizer
        if not _TEST_:
            with tf.name_scope('Train'):
                Optimizer_params_adam = {'beta1': 0.9, 'beta2':0.999, 'epsilon':0.1}
                Simple_DNN.Set_optimizer(starter_learning_rate= _LEARNING_RATE_, Optimizer='ADAM', Optimizer_params=Optimizer_params_adam, decay_steps=5000, Gradient_norm=5.0)

        #Construct op to check accuracy
        Simple_DNN.Construct_Accuracy_op()
        Simple_DNN.Construct_Predict_op()
        Simple_DNN.Set_test_control({'Dropout_prob_ph': 1, 'State': 'Train'})
        Simple_DNN.Set_train_control({'Dropout_prob_ph': 0.8, 'State': 'Train'})

        #Config block
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True


        if _TEST_:
            test_path = _TEST_PATH_
            test_images = os.listdir(test_path)
            test_image_path = tf.placeholder(tf.string)
            test_image = tf.image.decode_image(tf.read_file(test_image_path)) #Got Test image and pre-process
            test_image.set_shape([900, 900, 3])
            test_image = writer_pre_proc_seg_test(test_image)


    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())
        print('Global Vars initialized')
        if _TEST_:

            Simple_DNN.Construct_Writers()
            Simple_DNN.Try_restore()
            total_images = len(test_images)

            for index, imag in enumerate(test_images):
                printProgressBar(index+1, total_images)
                    test_imag = session.run([test_image], feed_dict={test_image_path:os.path.join(test_path,imag)})[0]
                    gt_name,_ = os.path.splitext(imag)
                    gt_name = gt_name + _GT_SUFFIX_
                    print(imag)
                    print(gt_name)
                    temp_mag = np.reshape(test_imag, (_IMAGE_HEIGHT_,_IMAGE_WIDTH_,3))
                    cv2.imshow('Test_imag',temp_mag)
                    cv2.waitKey(1)
                    prediction = Simple_DNN.Predict(test_imag, _PREDICTION_PATH_ + gt_name)
                    seg_map = construct_segmap(np.squeeze(prediction), gt_name) #Get Dice and Jacar'd coeff

                    print("DCs:" ,Dice_coeffs)
                    print("Mean Dice coeff:", np.mean(Dice_coeffs))
                    print("Mean Jacard coeff:", np.mean(Jacard_coeff))


        if not _TEST_:
            #Train
            print('Constructing Writers')
            Simple_DNN.Construct_Writers()
            print('Writers Constructed')
            Simple_DNN.Train_Iter(iterations=_ITERATIONS_, save_iterations=_SAVE_INTERVAL_, data=dummy_reader, restore=_RESTORE_)

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('       Progress: |%s| %s%% %s' % (bar, percent, suffix), end="\r")
    # Print New Line on Complete

if __name__ == "__main__":
    main()

