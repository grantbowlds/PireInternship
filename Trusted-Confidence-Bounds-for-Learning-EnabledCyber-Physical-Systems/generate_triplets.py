from itertools import permutations, combinations
import numpy as np

def class_separation(x_train,y_train):
    class_idxs=[]
    for data_class in sorted(set(y_train)):
        class_idxs.append(np.where((y_train == data_class))[0])
    return class_idxs

def generate_hard_positives_hard_negatives(x_train,y_train,idxs_per_class,samples_per_class,model):
    num_of_classes=len(set(y_train))
    samples=np.empty((num_of_classes,samples_per_class),dtype='int')
    #pick samples_per_class samples per class
    for data_class in sorted(set(y_train)):
        samples[data_class]=random.sample(list(idxs_per_class[data_class]),k=samples_per_class)

    Anchor=[]
    Positive=[]
    Negative=[]

    count=0
    for data_class in sorted(set(y_train)):
        print("class:",data_class)
        Embeddings_in=model.predict(x_train[samples[data_class]])

        different_classes_idxs=samples[np.arange(len(set(y_train)))!=data_class].flatten()
        Embeddings_out=model.predict(x_train[different_classes_idxs])


        for i in range(samples_per_class):
            Anchor_embedding=Embeddings_in[i]
            other_positives_idxs=np.arange(samples_per_class)[np.arange(samples_per_class)!=i]
            Positive_embeddings=Embeddings_in[other_positives_idxs]
            positive_distances=(Positive_embeddings-Anchor_embedding)**2
            positive_distances=np.sum(positive_distances,axis=1)
            hard_positive_idx=np.argmax(positive_distances)

            negative_distances=(Embeddings_out-Anchor_embedding)**2
            negative_distances=np.sum(negative_distances,axis=1)
            
            for j in range(len(different_classes_idxs)):
                if negative_distances[j]<positive_distances[hard_positive_idx]:
                    Anchor.append(x_train[samples[data_class,i]])
                    Positive.append(x_train[samples[data_class,other_positives_idxs[hard_positive_idx]]])
                    Negative.append(x_train[different_classes_idxs[j]])
        print(len(Anchor))


    Anchor=np.array(Anchor)
    Positive=np.array(Positive)
    Negative=np.array(Negative)

    return Anchor,Positive,Negative

def generate_random_triplets(data,ap_pairs_train,an_pairs_train,ap_pairs_test,an_pairs_test):
    train_xy = tuple([data['x_train'],data['y_train']])
    test_xy = tuple([data['x_validation'],data['y_validation']])

    triplet_train_pairs = []
    triplet_test_pairs = []

    #train
    for data_class in sorted(set(train_xy[1])):
        same_class_idx = np.where((train_xy[1] == data_class))[0]
        diff_class_idx = np.where(train_xy[1] != data_class)[0]
        A_P_pairs = random.sample(list(permutations(same_class_idx,2)),k=ap_pairs_train) #Generating Anchor-Positive pairs
        Neg_idx = random.sample(list(diff_class_idx),k=an_pairs_train)
        A_P_len = len(A_P_pairs)
        Neg_len = len(Neg_idx)
        for ap in A_P_pairs:
            Anchor = train_xy[0][ap[0]]
            Positive = train_xy[0][ap[1]]
            for n in Neg_idx:
                Negative = train_xy[0][n]
                triplet_train_pairs.append([Anchor,Positive,Negative])   

    #test
    for data_class in sorted(set(test_xy[1])):
        same_class_idx = np.where((test_xy[1] == data_class))[0]
        diff_class_idx = np.where(test_xy[1] != data_class)[0]
        A_P_pairs = random.sample(list(permutations(same_class_idx,2)),k=ap_pairs_test) #Generating Anchor-Positive pairs
        Neg_idx = random.sample(list(diff_class_idx),k=an_pairs_test)

        A_P_len = len(A_P_pairs)
        Neg_len = len(Neg_idx)
        for ap in A_P_pairs:
            Anchor = test_xy[0][ap[0]]
            Positive = test_xy[0][ap[1]]
            for n in Neg_idx:
                Negative = test_xy[0][n]
                triplet_test_pairs.append([Anchor,Positive,Negative])

    print(np.array(triplet_train_pairs).shape)
    return np.array(triplet_train_pairs), np.array(triplet_test_pairs)