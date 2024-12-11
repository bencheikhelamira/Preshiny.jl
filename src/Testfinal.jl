# Appel de la fonction pour obtenir la prédiction

include("/Users/pro/Desktop/m1-ssd/outilsR/appel_de_modele.jl")

# image_path = "/Users/pro/Desktop/m1-ssd/outilsR/resized_image1.png"
# model_checkpoint_path = "/Users/pro/Desktop/m1-ssd/outilsR/model-checkpoint.jld2"
function Testfinal(image_path::String ,model_checkpoint_path::String )

    predicted_class = predict_image_class(image_path, model_checkpoint_path)
    println("La classe prédite est : ", predicted_class)
end



image_path = "/Users/pro/Desktop/m1-ssd/outilsR/resized_image1.png"
model_checkpoint_path = "/Users/pro/Desktop/m1-ssd/outilsR/model-checkpoint.jld2"
predicted_class = predict_image_class(image_path, model_checkpoint_path)
println("La classe prédite est : ", predicted_class)