using Images, Flux
using Flux
using JLD2
using Images
using ImageTransformations
# include("./src/main.jl")
# #   L'ENTRAINEMENT ET LE SAUVEGARDE !!!!!!!
# # JE L'AI FAIT UNE SEULE FOIS  !!!!!!!!!
# # Créer le dataset
# x_train, y_train, x_test, y_test = createDataset()
# model=model3
# loss(x, y) = crossentropy(model(x), y)
# opt = Adam(0.001) 
# x_train = reshape(x_train, 28, 28, 1, :)
# x_test = reshape(x_test, 28, 28, 1, :)
# num_epochs = 10
# batch_size = 64
# num_images = 64
# batch_images = x_test[:, :, :, 1:num_images]
# #entrainement(x_train , y_train , x_test,y_test,model)
# # #creer une structure qui contient le modele et le fichier ou on va le charger 
# model_state = Flux.state(model);
# jldsave("mymodel.jld2"; model_state)
# # # sauvegarder le model ds chaque epoque .
# entrainement_cp(x_train , y_train , x_test,y_test,model)
# prediction_final(model3 , batch_images, num_images )



# Définir la fonction de prédiction
function predict_image_class(image_path::String, model_checkpoint_path::String)
    # Charger le modèle
    model =  Chain(
        Conv((5, 5), 1 => 8, relu),
        MaxPool((2, 2)),
        Dropout(0.25),             # Dropout après la première couche de pooling
        Conv((3, 3), 8 => 16, relu),
        MaxPool((2, 2)),
        Dropout(0.25),             # Dropout après la seconde couche de pooling
        Flux.flatten,
        Dense(5*5*16 => 64, relu),
        Dropout(0.5),              # Dropout après la première couche dense
        Dense(64 => 10),
        softmax
    )
    
    # Charger les paramètres du modèle sauvegardé
    model_state = JLD2.load(model_checkpoint_path, "model_state")
    
    # Fonction pour charger les paramètres dans le modèle
    function load_params!(model, model_state)
        for (i, layer) in enumerate(model.layers)
            if layer isa Conv
                layer.weight .= model_state.layers[i].weight
                layer.bias .= model_state.layers[i].bias
            elseif layer isa Dense
                layer.weight .= model_state.layers[i].weight
                layer.bias .= model_state.layers[i].bias
            end
        end
    end
    
    # Charger les paramètres dans le modèle
    load_params!(model, model_state)

    # Charger et prétraiter l'image
    img = Images.load(image_path)
    img_gray = Gray.(img)
    img_float64 = Float64.(img_gray) / 255.0
    img_reshaped = reshape(img_float64, (28, 28, 1, 1))

    # Effectuer la prédiction
    predictions = model(img_reshaped)
    predicted_class = argmax(predictions)[1] - 1
    
    # Retourner la classe prédite
    return predicted_class
end

