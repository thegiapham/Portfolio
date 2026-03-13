# app.R — merged, robust, and reactive

# library(shiny)
# library(rsconnect)
# library(RColorBrewer)
# library(shinythemes)
# library(DT)

library(readr)
library(dplyr)
library(stringr)
library(ggplot2)
library(broom)
library(car)
library(lmtest)
library(caret)
library(scales)
library(foreach)
# Vector-safe fallback operator
`%||%` <- function(a, b) { if (is.null(a) || length(a) == 0 || all(is.na(a))) b else a }

# ---- Load & prep data ----
df <- read.delim("housing-prices-ge19.txt", sep = "\t", header = TRUE)
names(df) <- str_replace_all(names(df), "\\.", "")
if (!"Price" %in% names(df)) stop("data.txt must contain a 'Price' column")
if ("Bathrooms" %in% names(df)) df <- df %>% filter(Bathrooms != 0)
df <- df %>% mutate(Price = log(Price))

if ("Test" %in% names(df)) {
  train_data <- subset(df, Test == 0)
} else {
  train_data <- df
}

candidate_vars <- names(train_data)[!names(train_data) %in% c("Price", "Test")]

mk_input <- function(id, var, data){
  x <- data[[var]]
  if (is.numeric(x)){
    rng <- range(x, na.rm = TRUE)
    step <- signif(diff(rng)/100, 2)
    numericInput(id, var, value = median(x, na.rm = TRUE),
                 min = floor(rng[1]), max = ceiling(rng[2]), step = step)
  } else {
    selectInput(id, var, var, choices = unique(as.character(x)))
  }
}

# ---- UI ----
ui <- navbarPage(
  title = "Predicting House Price",
  tabPanel(
    "Predict",
    sidebarLayout(
      sidebarPanel(
        actionButton("fit", "Fit / Predict", class = "btn-primary", style = "width: 100%;"),
        checkboxGroupInput("vars", "Select variables:", choices = candidate_vars,
                           selected = candidate_vars[1:min(3,length(candidate_vars))]),
        uiOutput("dyn_inputs"),
        sliderInput("conf", "Confidence level:", min = 0.80, max = 0.99, value = 0.95, step = 0.01)
      ),
      mainPanel(
        uiOutput("pred_html"),
        tableOutput("stat_table"),
        plotOutput("boxplot", height = 500)
      )
    )
  ),
  tabPanel(
    "Diagnostics",
    fluidRow(
      column(6, plotOutput("resid_plot", height = 320)),
      column(6, plotOutput("qq_plot", height = 320))
    ),
    uiOutput("vif_tab"),
    uiOutput("assump_tab")
  ),
  tabPanel(
    "Cross‑Validation",
    # Compact top toolbar (no wide sidebar)
    fluidRow(
      column(
        width = 12,
        div(
          style = "display:flex; gap:10px; align-items:center; margin:8px 0 12px; background:#f9fafb; padding:8px 12px; border-radius:6px;",
          tags$label("K-folds", `for`="kfold", style="margin:0; color:#6b7280;"),
          tags$input(
            type="number", id="kfold", value=5, min=3, max=10,
            class="form-control",
            style="width:90px; height:32px; padding:4px 6px; font-size:13px;"
          ),
          actionButton("run_cv","Run",
                       style="height:32px; padding:4px 12px; background-color:#374151; border:none; color:white; font-weight:600; border-radius:4px;")
        )
      )
    ),
    fluidRow(
      column(width = 12, uiOutput("cv_table"))
    )
  )
)

# ---- Server ----
server <- function(input, output, session){
  
  # State and frozen model configuration
  fitted_state <- reactiveVal(FALSE)
  model_vars   <- reactiveVal(character(0))   # predictors used in the current fitted model
  model_r      <- reactiveVal(NULL)           # fitted lm object
  
  k_folds <- reactive({ as.numeric(input$kfold) %||% 5 })
  
  # Fit/Reset: freeze predictors on fit, clear on reset
  observeEvent(input$fit, {
    if (!fitted_state()) {
      req(input$vars)
      model_vars(isolate(input$vars))
      f <- as.formula(paste("Price ~", paste(model_vars(), collapse = "+")))
      model_r(lm(f, data = train_data))
      fitted_state(TRUE)
    } else {
      fitted_state(FALSE)
      model_vars(character(0))
      model_r(NULL)
    }
  })
  
  observe({
    updateActionButton(session, "fit", label = if (fitted_state()) "Reset" else "Fit / Predict")
  })
  
  # Show inputs; when fitted, keep only the frozen predictors' inputs
  vars_for_inputs <- reactive({
    if (fitted_state()) model_vars() else (input$vars %||% character(0))
  })
  
  output$dyn_inputs <- renderUI({
    req(vars_for_inputs())
    lapply(vars_for_inputs(), function(v) mk_input(paste0("inp_", v), v, train_data))
  })
  
  # Newdata row built from frozen predictors
  newdata_row <- reactive({
    req(fitted_state())
    req(length(model_vars()) > 0)
    nd <- setNames(vector("list", length(model_vars())), model_vars())
    for (v in model_vars()) {
      val <- input[[paste0("inp_", v)]]
      if (is.numeric(train_data[[v]])) {
        nd[[v]] <- as.numeric(val)
      } else {
        tv <- train_data[[v]]
        nd[[v]] <- if (is.factor(tv)) factor(val, levels = levels(tv)) else as.character(val)
      }
    }
    as.data.frame(nd, stringsAsFactors = FALSE)
  })
  
  # Duan smearing factor
  smearing_factor <- reactive({
    m <- model_r(); req(m)
    mean(exp(residuals(m)), na.rm = TRUE)
  })
  
  # Live prediction after fit (no re-fit on value edits)
  predicted_vals <- reactive({
    req(fitted_state())
    m  <- model_r(); req(m)
    nd <- newdata_row(); req(nrow(nd) > 0)
    pr_conf_log <- predict(m, newdata = nd, interval = "confidence", level = input$conf)
    pr_pred_log <- predict(m, newdata = nd, interval = "prediction", level = input$conf)
    s_hat <- smearing_factor()
    list(conf = exp(pr_conf_log) * s_hat,
         pred = exp(pr_pred_log) * s_hat)
  })
  
  # ---- OUTPUTS ----
  
  output$pred_html <- renderUI({
    pr <- predicted_vals()
    HTML(sprintf(
      "<h3>Predicted Price: %s</h3>
       <p><strong>Confidence Interval (%.0f%%)</strong>: (%s, %s)<br>
       <strong>Prediction Interval (%.0f%%)</strong>: (%s, %s)</p>",
      dollar(pr$conf[1, 'fit']),
      input$conf*100, dollar(pr$conf[1, 'lwr']), dollar(pr$conf[1, 'upr']),
      input$conf*100, dollar(pr$pred[1, 'lwr']), dollar(pr$pred[1, 'upr'])
    ))
  })
  
  output$stat_table <- renderTable({
    req(fitted_state())
    m <- model_r(); req(m)
    y_log <- train_data$Price
    yhat_log <- fitted(m)
    s_hat <- smearing_factor()
    y_dol <- exp(y_log)
    yhat_dol <- exp(yhat_log) * s_hat
    tibble::tibble(
      Metric = c("AIC","RMSE (log)","MAE (log)","R² (log)","RMSE ($)","MAE ($)"),
      Value  = c(AIC(m),
                 sqrt(mean((y_log - yhat_log)^2, na.rm = TRUE)),
                 mean(abs(y_log - yhat_log), na.rm = TRUE),
                 summary(m)$r.squared,
                 sqrt(mean((y_dol - yhat_dol)^2, na.rm = TRUE)),
                 mean(abs(y_dol - yhat_dol), na.rm = TRUE))
    )
  })
  
  # Actual vs Predicted (thousand $) with focused zoom
  output$boxplot <- renderPlot({
    req(fitted_state())
    m <- model_r(); req(m)
    s_hat <- smearing_factor()
    plot_data <- train_data %>%
      mutate(predicted_price = exp(predict(m, newdata = train_data)) * s_hat,
             actual_price    = exp(Price))
    ggplot(plot_data, aes(x = predicted_price/1000, y = actual_price/1000)) +
      geom_point(alpha = 0.6) +
      geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
      coord_cartesian(xlim = c(0, 1000), ylim = c(0, 800)) +  # 0–1000k, 0–700k zoom
      theme_minimal() +
      labs(title = "Actual vs Predicted Prices (thousand $)",
           x = "Predicted Price (thousand $)", y = "Actual Price (thousand $)")
  })
  
  output$resid_plot <- renderPlot({
    req(fitted_state())
    m <- model_r(); req(m)
    ggplot(broom::augment(m), aes(.fitted, .resid)) +
      geom_point(alpha=0.6, size=2, color="black") +
      geom_hline(yintercept=0, lty=2, color="red", linewidth=1) +
      geom_smooth(method="loess", color="darkred", fill="lightcoral", alpha=0.2, se=TRUE) +
      theme_minimal() +
      labs(title = "Residuals vs Fitted Values (log-scale)",
           x = "Fitted (log-price)", y = "Residuals (log)")
  })
  
  output$qq_plot <- renderPlot({
    req(fitted_state())
    m <- model_r(); req(m)
    resids <- resid(m)
    qq_data <- qqnorm(resids, plot.it = FALSE)
    y_q <- quantile(resids, c(0.25, 0.75), na.rm = TRUE)
    x_q <- qnorm(c(0.25, 0.75))
    slope <- (y_q[2] - y_q[1]) / (x_q[2] - x_q[1])
    intercept <- y_q[1] - slope * x_q[1]
    df <- data.frame(x=qq_data$x, y=qq_data$y)
    ggplot(df, aes(x, y)) +
      geom_point(alpha=0.6, size=2, color="black") +
      geom_abline(slope=slope, intercept=intercept, color="red", linewidth=1) +
      theme_minimal() +
      labs(title = "Normal Q-Q Plot (log residuals)",
           x = "Theoretical Quantiles", y = "Sample Quantiles")
  })
  
  output$vif_tab <- renderUI({
    req(fitted_state())
    m <- model_r(); req(m)
    vif_vals <- tryCatch(car::vif(m), error=function(e) NA)
    if (all(is.na(vif_vals))) return(NULL)
    vif_df <- tibble::tibble(Variable = names(vif_vals), VIF = as.numeric(vif_vals)) %>%
      mutate(
        Color = case_when(VIF < 5 ~ "#059669", VIF < 10 ~ "#d97706", TRUE ~ "#dc2626"),
        Status = case_when(VIF < 5 ~ "Good", VIF < 10 ~ "Moderate", TRUE ~ "High")
      )
    html_table <- paste0(
      "<div style='margin-top:30px; margin-bottom:30px; padding:20px; background-color:#ffffff; border-radius:10px; border:1px solid #e5e7eb; box-shadow:0 2px 8px rgba(0,0,0,0.08);'>",
      "<h4 style='margin-top:0; margin-bottom:20px; color:#1f2937; font-weight:600; font-size:16px;'>Variance Inflation Factor (VIF)</h4>",
      "<table style='width:100%; border-collapse:collapse;'>",
      "<tr style='background-color:#374151; color:white;'><th style='padding:12px 15px; text-align:left;'>Variable</th><th style='padding:12px 15px; text-align:center;'>VIF Value</th><th style='padding:12px 15px; text-align:center;'>Status</th></tr>"
    )
    for (i in 1:nrow(vif_df)) {
      html_table <- paste0(html_table,
                           "<tr style='border-bottom:1px solid #e5e7eb; background-color:", ifelse(i %% 2 == 0, "#f9fafb", "#ffffff"), ";'>",
                           "<td style='padding:12px 15px; color:#1f2937; font-weight:500;'>", vif_df$Variable[i], "</td>",
                           "<td style='padding:12px 15px; text-align:center; color:#374151;'>", round(vif_df$VIF[i], 3), "</td>",
                           "<td style='padding:12px 15px; text-align:center;'><span style='background-color:", vif_df$Color[i], "; color:white; padding:6px 12px; border-radius:6px; font-weight:600; font-size:13px;'>", vif_df$Status[i], "</span></td></tr>")
    }
    html_table <- paste0(html_table, "</table>",
                         "<p style='font-size:13px; color:#6b7280; margin-top:15px; margin-bottom:0;'><strong>Guidelines:</strong> VIF < 5 is good, 5-10 is moderate, > 10 indicates multicollinearity.</p></div>")
    HTML(html_table)
  })
  
  # Cross-validation (sequential in eventReactive; UI only formats)
  cv_results <- eventReactive(input$run_cv, {
    req(fitted_state())
    req(input$vars)
    k <- as.integer(isolate(input$kfold) %||% 5)
    validate(need(k >= 2, "K must be >= 2"))
    validate(need(k <= nrow(train_data), "K must not exceed the number of rows in training data"))
    
    set.seed(22)
    folds <- caret::createFolds(seq_len(nrow(train_data)), k = k, returnTrain = FALSE)
    
    f <- as.formula(paste("Price ~", paste(isolate(input$vars), collapse = "+")))
    
    fold_metrics <- lapply(seq_along(folds), function(i){
      valid_idx <- folds[[i]]
      train_idx <- setdiff(seq_len(nrow(train_data)), valid_idx)
      
      fit_i <- lm(f, data = train_data[train_idx, , drop = FALSE])
      
      # Duan smearing estimated on training fold residuals
      s_hat_i <- mean(exp(residuals(fit_i)), na.rm = TRUE)
      
      # Predict on validation fold, back-transform to $
      pred_log  <- predict(fit_i, newdata = train_data[valid_idx, , drop = FALSE])
      pred_dol  <- exp(pred_log) * s_hat_i
      y_dol     <- exp(train_data$Price[valid_idx])
      
      rmse_i <- sqrt(mean((pred_dol - y_dol)^2, na.rm = TRUE))
      r2_i   <- 1 - sum((pred_dol - y_dol)^2, na.rm = TRUE) /
        sum((y_dol - mean(y_dol, na.rm = TRUE))^2, na.rm = TRUE)
      
      data.frame(Fold = paste0("Fold ", i),
                 RMSE = rmse_i,
                 Rsquared = r2_i,
                 stringsAsFactors = FALSE)
    })
    
    resample_data <- do.call(rbind, fold_metrics)
    
    list(
      resample = resample_data,
      rmse = c(mean = mean(resample_data$RMSE), sd = sd(resample_data$RMSE),
               min = min(resample_data$RMSE), max = max(resample_data$RMSE)),
      r2   = c(mean = mean(resample_data$Rsquared), sd = sd(resample_data$Rsquared),
               min = min(resample_data$Rsquared), max = max(resample_data$Rsquared))
    )
  }, ignoreInit = TRUE)
  
  output$cv_table <- renderUI({
    req(cv_results())
    res <- cv_results()
    resample_data <- res$resample
    
    rmse_stats <- res$rmse
    r2_stats   <- res$r2
    
    mean_rmse <- as.numeric(rmse_stats["mean"])
    sd_rmse   <- as.numeric(rmse_stats["sd"])
    min_rmse  <- as.numeric(rmse_stats["min"])
    max_rmse  <- as.numeric(rmse_stats["max"])
    
    mean_r2 <- as.numeric(r2_stats["mean"])
    sd_r2   <- as.numeric(r2_stats["sd"])
    min_r2  <- as.numeric(r2_stats["min"])
    max_r2  <- as.numeric(r2_stats["max"])
    
    summary_cards <- paste0(
      "<div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px;'>",
      "<div style='padding: 20px; background-color: #f3f4f6; border-radius: 10px; border-left: 4px solid #374151; text-align: center;'>",
      "<p style='margin: 0; color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase;'>Mean RMSE ($)</p>",
      "<p style='margin: 10px 0 0 0; color: #1f2937; font-size: 24px; font-weight: 700;'>", round(mean_rmse, 2), "</p>",
      "</div>",
      "<div style='padding: 20px; background-color: #ecfdf5; border-radius: 10px; border-left: 4px solid #059669; text-align: center;'>",
      "<p style='margin: 0; color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase;'>Min RMSE ($)</p>",
      "<p style='margin: 10px 0 0 0; color: #059669; font-size: 24px; font-weight: 700;'>", round(min_rmse, 2), "</p>",
      "</div>",
      "<div style='padding: 20px; background-color: #fef2f2; border-radius: 10px; border-left: 4px solid #dc2626; text-align: center;'>",
      "<p style='margin: 0; color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase;'>Max RMSE ($)</p>",
      "<p style='margin: 10px 0 0 0; color: #dc2626; font-size: 24px; font-weight: 700;'>", round(max_rmse, 2), "</p>",
      "</div>",
      "<div style='padding: 20px; background-color: #fef3c7; border-radius: 10px; border-left: 4px solid #d97706; text-align: center;'>",
      "<p style='margin: 0; color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase;'>SD RMSE ($)</p>",
      "<p style='margin: 10px 0 0 0; color: #d97706; font-size: 24px; font-weight: 700;'>", round(sd_rmse, 2), "</p>",
      "</div>",
      "</div>"
    )
    
    r2_cards <- paste0(
      "<div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 30px;'>",
      "<div style='padding: 20px; background-color: #f3f4f6; border-radius: 10px; border-left: 4px solid #374151; text-align: center;'>",
      "<p style='margin: 0; color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase;'>Mean R² ($-scale)</p>",
      "<p style='margin: 10px 0 0 0; color: #1f2937; font-size: 24px; font-weight: 700;'>", round(mean_r2, 4), "</p>",
      "</div>",
      "<div style='padding: 20px; background-color: #fef2f2; border-radius: 10px; border-left: 4px solid #dc2626; text-align: center;'>",
      "<p style='margin: 0; color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase;'>Min R²</p>",
      "<p style='margin: 10px 0 0 0; color: #dc2626; font-size: 24px; font-weight: 700;'>", round(min_r2, 4), "</p>",
      "</div>",
      "<div style='padding: 20px; background-color: #ecfdf5; border-radius: 10px; border-left: 4px solid #059669; text-align: center;'>",
      "<p style='margin: 0; color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase;'>Max R²</p>",
      "<p style='margin: 10px 0 0 0; color: #059669; font-size: 24px; font-weight: 700;'>", round(max_r2, 4), "</p>",
      "</div>",
      "<div style='padding: 20px; background-color: #fef3c7; border-radius: 10px; border-left: 4px solid #d97706; text-align: center;'>",
      "<p style='margin: 0; color: #6b7280; font-size: 12px; font-weight: 600; text-transform: uppercase;'>SD R²</p>",
      "<p style='margin: 10px 0 0 0; color: #d97706; font-size: 24px; font-weight: 700;'>", round(sd_r2, 4), "</p>",
      "</div>",
      "</div>"
    )
    
    # Fold-by-fold table
    fold_table <- paste0(
      "<div style='margin-top: 30px; padding: 20px; background-color: #ffffff; border-radius: 10px; border: 1px solid #e5e7eb; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>",
      "<h4 style='margin-top: 0; margin-bottom: 20px; color: #1f2937; font-weight: 600; font-size: 16px;'>Fold-by-Fold Results ($-scale RMSE)</h4>",
      "<table style='width: 100%; border-collapse: collapse;'>",
      "<tr style='background-color: #374151; color: white;'>",
      "<th style='padding: 12px 15px; text-align: left; font-weight: 600; border: none;'>Fold</th>",
      "<th style='padding: 12px 15px; text-align: center; font-weight: 600; border: none;'>RMSE ($)</th>",
      "<th style='padding: 12px 15px; text-align: center; font-weight: 600; border: none;'>R²</th>",
      "</tr>"
    )
    for (i in 1:nrow(resample_data)) {
      fold_table <- paste0(
        fold_table,
        "<tr style='border-bottom: 1px solid #e5e7eb; background-color: ", ifelse(i %% 2 == 0, "#f9fafb", "#ffffff"), ";'>",
        "<td style='padding: 12px 15px; color: #1f2937; font-weight: 500;'>", resample_data$Fold[i], "</td>",
        "<td style='padding: 12px 15px; text-align: center; color: #374151;'>", round(resample_data$RMSE[i], 2), "</td>",
        "<td style='padding: 12px 15px; text-align: center; color: #374151;'>", round(resample_data$Rsquared[i], 4), "</td>",
        "</tr>"
      )
    }
    fold_table <- paste0(
      fold_table, "</table>",
      "<p style='font-size: 13px; color: #6b7280; margin-top: 15px; margin-bottom: 0;'>",
      "<strong>Note:</strong> CV fits the model on log(Price) each fold but reports errors on $ by exponentiating predictions and applying a fold‑specific Duan smearing factor.",
      "</p></div>"
    )
    
    HTML(paste0(summary_cards, r2_cards, fold_table))
  })
}

shinyApp(ui, server)
