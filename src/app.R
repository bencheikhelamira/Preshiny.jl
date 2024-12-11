library(shiny)
library(imager)
library(base64enc)
require(Rulia)
# UI : Interface utilisateur
ui <- fluidPage(
  titlePanel("Dessin d'un Chiffre"),
  
  sidebarLayout(
    sidebarPanel(
      actionButton("clear_btn", "Effacer"),
      actionButton("save_btn", "Sauvegarder")
    ),
    
    mainPanel(
      # Zone de dessin
      tags$canvas(id = "draw_area", width = 28, height = 28, style = "border:1px solid #000000; background-color: #fff;"),
      br(),
      textOutput("instructions"),
      tags$script(HTML("
        // Initialisation du canevas et du contexte pour dessiner
        let canvas = document.getElementById('draw_area');
        let context = canvas.getContext('2d');
        let painting = false;

        // Fonction pour démarrer le dessin
        function startPosition(e) {
          painting = true;
          draw(e);
        }

        // Fonction pour arrêter le dessin
        function endPosition() {
          painting = false;
          context.beginPath();
        }

        // Fonction pour dessiner
        function draw(e) {
          if (!painting) return;
          context.lineWidth = 1;
          context.lineCap = 'round';
          context.strokeStyle = 'pink';
          
          context.lineTo(e.offsetX, e.offsetY);
          context.stroke();
          context.beginPath();
          context.moveTo(e.offsetX, e.offsetY);
        }

        // Associer les événements de la souris au canevas
        canvas.addEventListener('mousedown', startPosition);
        canvas.addEventListener('mouseup', endPosition);
        canvas.addEventListener('mousemove', draw);

        // Capturer l'image et l'envoyer à R pour sauvegarder
        document.getElementById('save_btn').onclick = function() {
          let dataURL = canvas.toDataURL('image/png');
          Shiny.setInputValue('img_data', dataURL, {priority: 'event'});
        };

        // Effacer le canevas
        document.getElementById('clear_btn').onclick = function() {
          context.clearRect(0, 0, canvas.width, canvas.height);
        };
      "))
    )
  )
)

server <- function(input, output, session) {
  
  # Créer une variable réactive pour stocker l'image sauvegardée
  saved_img_path <- reactiveVal(NULL)
  
  # Observer l'image dessinée par l'utilisateur pour la sauvegarder
  observeEvent(input$img_data, {
    if (!is.null(input$img_data)) {
      # Extraire la partie base64 de l'image (enlever le préfixe 'data:image/png;base64,')
      img_data <- sub("data:image/png;base64,", "", input$img_data)
      
      # Décoder l'image base64 en binaire
      img_raw <- base64decode(img_data)
      
      # Créer un fichier temporaire pour sauvegarder l'image
      temp_img_path <- tempfile(fileext = ".png")
      writeBin(img_raw, temp_img_path)
      
      # Charger l'image avec imager
      img <- load.image(temp_img_path)
      
      # Redimensionner l'image à 28x28 pixels
      img_resized <- imresize(img, 28, 28)
      
      # Créer un fichier temporaire pour sauvegarder l'image redimensionnée
      resized_img_path <- tempfile(fileext = ".png")
      
      # Sauvegarder l'image redimensionnée
      save.image(img_resized, resized_img_path)
      
      # Mettre à jour la variable réactive avec le chemin de l'image redimensionnée
      saved_img_path(resized_img_path)
      
      # Afficher le chemin du fichier sauvegardé pour vérifier
      output$instructions <- renderText({
        paste("Image redimensionnée sauvegardée à :", resized_img_path)
      })
    }
  })
  
  # Observer le bouton de sauvegarde pour lancer la prédiction
  observeEvent(input$save_btn, {
    # Vérifier si l'image redimensionnée est sauvegardée et l'utiliser pour la prédiction
    if (!is.null(saved_img_path())) {
      img_path <- saved_img_path()  # Récupérer le chemin de l'image sauvegardée
      print(paste("Chemin de l'image sauvegardée :", img_path))
      
      # Appeler la fonction pour traiter l'image et la passer au modèle
      save_path <- tempfile(fileext = ".png")  # Créer un autre chemin temporaire pour la prédiction
      process_image(img_path, 28, 28, save_path)
      
      # Appeler la fonction de prédiction en Julia
      jl(`
         include("/Users/pro/Desktop/m1-ssd/outilsR/Testfinal.jl")
         image_path = "${save_path}"
         model_checkpoint_path = "/Users/pro/Desktop/m1-ssd/outilsR/model-checkpoint.jld2"
         predicted_class = predict_image_class(image_path, model_checkpoint_path)
         println("La classe prédite est : ", predicted_class)
         `)
    }
  })
}

# Lancer l'application
shinyApp(ui = ui, server = server)
