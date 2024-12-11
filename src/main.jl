using Images
using MLDatasets
using ImageView
using Flux 

#using Flux: onehotbatch, argmax, crossentropy, throttle
using Base.Iterators: repeated
using Images
using Flux: onehotbatch, onecold, crossentropy , params

#Préparation des données : 
function load_prepare_data()

    # Charger le dataset MNIST
    train_X, train_y = MNIST.traindata()
    test_X, test_y = MNIST.testdata()

    #Normaliser entre 0 et 1 : 
    #divisant par 255 pour améliorer la convergence de l'apprentissage
    train_X = Float32.(train_X) ./ 255.0
    test_X = Float32.(test_X) ./ 255.0

    #Ajouter une dimension pour les canaux , car il est utile après pour notre modele de réseaux de neurones
    train_X = reshape(train_X, 28, 28, 1, :)
    test_X = reshape(test_X, 28, 28, 1, :)

    # Encodage one-hot des labels: Transformer les labels en vecteurs binaires pour faciliter l'apprentissage
    train_y = onehotbatch(train_y, 0:9)
    test_y = onehotbatch(test_y, 0:9)

    return train_X, train_y, test_X, test_y
end

#Construction du modèle CNN: 
function build_cnn_model()

    model = Chain(
        Conv((3, 3), 1 => 16, relu),    # Convolution 3x3, 1 canal d'entrée (image en niveaux de gris) et 16 filtres
        MaxPool((2, 2)),                # Max pooling pour réduire la taille de l'image
        Conv((3, 3), 16 => 32, relu),   # Deuxième couche convolutionnelle
        MaxPool((2, 2)),                # Deuxième max pooling
        Flux.flatten,                   # Aplatir la sortie des convolutions pour la passer dans les couches denses
        Dense(32 * 5 * 5 => 128, relu), # Couche dense après le flatten, ajustée en fonction de la taille de l'image
        Dense(128 => 10),               # Couche de sortie avec 10 neurones pour les 10 classes
        softmax                         # Softmax pour obtenir les probabilités des classes
    )
    return model
end


#************* Entrainement******************

# Fonction pour calculer l'accuracy
#Elle prédit les classes pour chaque lot, puis compare ces prédictions avec les étiquettes réelles
function compute_accuracy(loader, model)
    accuracy = 0.0
    total = 0
    for (x_batch, y_batch) in loader
        predict = model(x_batch)                        # Prédire les classes
        predicted_classes = onecold(predict, 0:9)       # Convertir les prédictions en classes
        true_labels = onecold(y_batch, 0:9)             # Convertir les étiquettes réelles en classes
        accuracy += sum(predicted_classes .== true_labels)
        total += length(true_labels)
    end
    return accuracy / total
end

#the testing data should be used to measure the accuracy of the model.

function train_model(train_loader, test_loader, model, epochs, optimizer)

    loss(x, y) = crossentropy(model(x), y)
    for epoch in 1:epochs
        for (x_batch, y_batch) in train_loader
            Flux.train!(loss, params(model), [(x_batch, y_batch)], optimizer)
        end
        test_accuracy = compute_accuracy(test_loader, model)
        println("Époque $epoch terminée. Précision sur le test : $test_accuracy")
    end
       
end

"""
Le recall : Plus il est élevé, plus le modèle de Machine Learning maximise le nombre de Vrai Positif, il 
ne ratera aucun positif.

La précision: Quand la précision est haute, cela veut dire que la majorité des prédictions positives du modèle 
sont des positifs bien prédit, il minimise le nombre de faux positifs.
Plus la precision est haute, moins le modèle se trompe sur les positifs 
F1- SCORE: Plus le F1 Score est élevé, plus le modèle est performant.

True Negative (TN) : la prédiction et la valeur réelle sont négatives. 
True Positive (TP) : la prédiction et la valeur réelle sont positives. 
False Positive (FP) : la prédiction est positive alors que la valeur réelle est négative.
False Negative (FN) : la prédiction est négative alors que la valeur réelle est négative.
"""
function metriques(predicted_classes, true_labels)
    tp = sum((predicted_classes .== true_labels) .& (true_labels .== 1))  # Vrais positifs
    fp = sum((predicted_classes .== 1) .& (true_labels .== 0))            # Faux positifs
    fn = sum((predicted_classes .== 0) .& (true_labels .== 1))            # Faux négatifs

    precision = tp / (tp + fp + 1e-10)  # Ajout d'une petite constante pour éviter la division par zéro
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    return precision, recall, f1
end

#ces 2 fonctions portent le meme nom mais avec des parametres différents , c'est ça ce qu'on appelle du multiple dispatching
#ici , la 2eme fonction "metriques" calcule la matrice de confusion qui s'appelle aussi un tableau de contingence, 
#elle permet de visualiser la capacité du modèle à classer correctement ou incorrectement les cas en comparant les prédictions du modèle avec la vérité réelle.
#La matrice de confusion est une table qui compare les classes prédites avec les classes réelles pour comprendre où le modèle se trompe le plus
function metriques(predicted_classes, true_labels, num_classes)
    cm = zeros(Int, num_classes, num_classes)
    for (pred, true_label) in zip(predicted_classes, true_labels)
        cm[true_label + 1, pred + 1] += 1  # +1 car les indices en Julia commencent à 1
    end
    return cm
end


function display_metriques(test_loader, model)

    for (x_batch, y_batch) in test_loader  # Utilise les données de test
        predicted_classes = onecold(model(x_batch), 0:9)  # Prédire les classes pour ce mini-lot
        true_labels = onecold(y_batch, 0:9)               # Convertir les étiquettes réelles en classes

        # Calculer les métriques pour ce mini-lot
        precision, recall, f1 = metriques(predicted_classes, true_labels)
        println("Précision: ", precision)
        println("Recall: ", recall)
        println("F1-score: ", f1)

        # Calculer la matrice de confusion pour ce mini-lot
        cm = metriques(predicted_classes, true_labels, 10)  # 10 classes pour MNIST
        println("Matrice de confusion:")
        println(cm)

        break  # Supprimer cette ligne pour parcourir tous les batches du test_loader
    end
end

"""
REMARQUES : 

    Interpretation de la matrice de confusion: 
    * 6 échantillons de la classe 0 ont été correctement classés comme 0 
    * 10 échantillions de la classe 1 ont été correctement classés comme 1
    * Ainsi de suite ...
    Analyse des métriques ( précision, recall , f1-score):
    * on a 0.9999999 pour précision , recall et aussi f1-score, ce qui montre que notre modele est performant
    Le modèle fait un excellent travail pour distinguer les différentes classes, avec une précision proche de 100 % pour plusieurs classes.
    Il n’y a pratiquement pas de confusion entre les classes. Cette matrice reflète bien les bons résultats qu'on a obtenu pour les autres métriques comme le F1-score, la précision, et le rappel.

Cela suggère que notre modèle est très performant pour la tâche de classification des images MNIST.

"""

function main()
    # Charger les données
    train_X, train_y, test_X, test_y = load_prepare_data()

    # Construire le modèle
    model = build_cnn_model()

    # Définir les loaders
    train_loader = Flux.DataLoader((train_X, train_y), batchsize=64, shuffle=true)
    test_loader = Flux.DataLoader((test_X, test_y), batchsize=64)

    # Optimiseur et paramètres d'entraînement
    optimizer = Adam()
    epochs = 10

    # Entraîner le modèle
    train_model(train_loader, test_loader, model, epochs, optimizer)

    # Calcul des métriques après l'entraînement
    total_accuracy = compute_accuracy(test_loader, model)
    println("Précision totale sur le set de test: $total_accuracy")

    display_metriques(test_loader, model)
end

# Appeler la fonction principale
main()
