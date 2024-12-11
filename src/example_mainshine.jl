
include("mainshine.jl")


# Charger le dataset MNIST
train_X, train_y = MNIST.traindata()
test_X, test_y = MNIST.testdata()

# Charger les données
train_X, train_y, test_X, test_y = load_prepare_data(train_X, train_y, test_X, test_y)

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

# Enregistrer la matrice de confusion finale
for (x_batch, y_batch) in test_loader
    predicted_classes = onecold(model(x_batch), 0:9)
    true_labels = onecold(y_batch, 0:9)
    cm = metriques(predicted_classes, true_labels, 10)
    class_labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    saveConfusionMatrixToCSV(cm, class_labels)  # Sauvegarde dans CSV
    println("Matrice de confusion enregistrée dans confusion_matrix.csv")
    break
end


