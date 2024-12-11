library(shiny)
library(ggplot2)
library(reshape2)
library(dplyr)
library(tidyr)

# Define UI for application
ui <- fluidPage(
  titlePanel("Tableau de Bord des Performances du Modèle CNN"),
  sidebarLayout(
    sidebarPanel(
      h3("Métriques de Performance"),
      textOutput("precision"),
      textOutput("recall"),
      textOutput("f1_score")
    ),
    mainPanel(
      h3("Matrice de Confusion"),
      plotOutput("confusionMatrixPlot"),
      h3("Performance par Classe"),
      plotOutput("classPerformancePlot")
    )
  )
)

server <- function(input, output) {
  # Charger les données de performance
  metrics <- read.csv("model_metrics.csv", header = TRUE)
  
  # Utiliser la dernière ligne pour les métriques finales
  final_metrics <- metrics[nrow(metrics), ]
  
  # Charger la matrice de confusion et retirer la dernière colonne "Class"
  confusion_data <- read.csv("confusion_matrix.csv", header = TRUE)
  confusion_matrix <- as.matrix(confusion_data[, -11])  # Exclut la colonne "Class"
  rownames(confusion_matrix) <- colnames(confusion_matrix) <- 0:9  # Assigne les noms de lignes et colonnes
  
  # Affichage des métriques
  output$precision <- renderText({ paste("Précision :", round(final_metrics$Precision, 2)) })
  output$recall <- renderText({ paste("Rappel :", round(final_metrics$Recall, 2)) })
  output$f1_score <- renderText({ paste("F1-Score :", round(final_metrics$F1, 2)) })
  
  # Graphique de la matrice de confusion
  output$confusionMatrixPlot <- renderPlot({
    cm_melt <- melt(confusion_matrix)
    ggplot(cm_melt, aes(Var1, Var2, fill = value)) +
      geom_tile(color = "white") +
      scale_fill_gradient(low = "white", high = "blue") +
      geom_text(aes(label = round(value, 1)), color = "black", size = 4) +
      scale_x_continuous(breaks = 0:9, labels = 0:9) +  # Échelle de 0 à 9 pour l'axe x
      scale_y_continuous(breaks = 0:9, labels = 0:9) +  # Échelle de 0 à 9 pour l'axe y
      labs(x = "Classe Prédite", y = "Classe Réelle", title = "Matrice de Confusion") +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12)
      )
  })
  
  # Graphique de la performance par classe
  output$classPerformancePlot <- renderPlot({
    class_performance <- metrics %>%
      pivot_longer(cols = c(Precision, Recall, F1), names_to = "Métrique", values_to = "Valeur")
    
    ggplot(class_performance, aes(x = Epoch, y = Valeur, color = Métrique)) +
      geom_line(size = 1.2) +
      geom_point(size = 2) +
      scale_x_continuous(breaks = seq(1, 10, by = 1)) +  # Forcer un pas de 1 sur l'axe des époques
      scale_y_continuous(limits = c(0.95, 1)) +
      labs(x = "Époque", y = "Valeur", title = "Performance au Fil des Époques") +
      theme_minimal() +
      scale_color_manual(values = c("Precision" = "blue", "Recall" = "green", "F1" = "red")) +
      theme(
        plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12),
        legend.position = "bottom"
      )
  })
  
}

# Run the application 
shinyApp(ui = ui, server = server)
