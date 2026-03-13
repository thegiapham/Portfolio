# ── Packages ─────────────────────────────────────────────────────────────────
library(shiny)
library(readr)
library(dplyr)
library(stringr)
library(ggplot2)
library(broom)
library(car)
library(lmtest)
library(caret)
library(scales)
library(bslib)
library(shinyWidgets)
library(DT)
library(plotly)
library(shinycssloaders)

options(scipen = 999)

# ── Data ─────────────────────────────────────────────────────────────────────
df <- read.delim("housing-prices-ge19.txt")
names(df) <- str_replace_all(names(df), "\\.", "")

candidate_vars <- names(df)[!names(df) %in% c("Price", "Test")]
if (!"Price" %in% names(df)) stop("data.txt must contain a 'Price' column")

train_data <- if ("Test" %in% names(df)) subset(df, Test == 0) else df

fmt_dollar <- function(x) dollar(x, accuracy = 1)

mk_input <- function(id, var, data){
  x <- data[[var]]
  if (is.numeric(x)){
    rng <- range(x, na.rm = TRUE)
    step <- max(1, signif(diff(rng)/100, 2))
    numericInput(
      id, label = var,
      value = median(x, na.rm = TRUE),
      min = floor(rng[1]), max = ceiling(rng[2]), step = step
    )
  } else {
    choices <- unique(as.character(x))
    pickerInput(
      inputId = id, label = var,
      choices = choices, selected = choices[1],
      multiple = FALSE, options = list(`live-search` = TRUE)
    )
  }
}

# ── Theme ────────────────────────────────────────────────────────────────────
app_theme <- bs_theme(
  version = 5,
  bootswatch = "minty",
  base_font = font_google("Inter"),
  heading_font = font_google("Inter Tight"),
  primary = "#2563eb",
  secondary = "#14b8a6"
)

# ── UI ───────────────────────────────────────────────────────────────────────
ui <- navbarPage(
  title = div("House Price Lab", style = "font-weight:700;"),
  theme = app_theme,
  header = tagList(
    tags$style(HTML("
      .app-gradient {
        background: linear-gradient(90deg, rgba(37,99,235,0.12), rgba(20,184,166,0.12));
        border-radius: 14px; padding: 14px 18px; margin-bottom: 16px;
      }
      .kpi { border-radius: 14px; padding: 16px; color: #0f172a; box-shadow: 0 6px 16px rgba(2,6,23,0.08); }
      .kpi h4 { margin: 0 0 4px 0; font-weight: 700; }
      .kpi .val { font-size: 1.4rem; font-weight: 800; }
      .kpi-aic { background: #e0e7ff; }
      .kpi-rmse{ background: #fee2e2; }
      .kpi-r2  { background: #dcfce7; }
      .pred-card { border-radius: 16px; padding: 16px 18px; background: white;
                   box-shadow: 0 8px 22px rgba(2,6,23,0.08); }
      .badge { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700; }
      .badge-ci { background:#e0f2fe; color:#0369a1; }
      .badge-pi { background:#fef3c7; color:#92400e; }
      .btn-primary { border-radius: 999px; padding: 8px 14px; font-weight:700; }
      .dataTables_wrapper .dataTables_filter input { border-radius:12px; }
    "))
  ),
  tabPanel(
    "Predict",
    sidebarLayout(
      sidebarPanel(
        div(class="app-gradient",
            HTML("<b>Build your model</b><br><small>Select predictors and enter values</small>")
        ),
        pickerInput(
          "vars", "Select variables:",
          choices = candidate_vars,
          selected = candidate_vars[seq_len(min(3, length(candidate_vars)))],
          multiple = TRUE,
          options = list(
            `actions-box` = TRUE,
            `live-search` = TRUE,
            `selected-text-format` = "count > 3"
          )
        ),
        uiOutput("dyn_inputs"),
        sliderInput("conf", "Confidence level:", min = 0.80, max = 0.99, value = 0.95, step = 0.01),
        prettySwitch("interactive", "Interactive plots (Plotly)", value = TRUE, status = "primary"),
        actionBttn("fit", "Fit / Predict", style = "fill", color = "primary", size = "md", icon = icon("wand-magic-sparkles"))
      ),
      mainPanel(
        uiOutput("pred_card"),
        br(),
        uiOutput("kpi_cards"),
        br(),
        h4("Model Statistics"), 
        DTOutput("stat_table"),
        br(),
        h4("Price vs First Selected Variable"),
        withSpinner(plotlyOutput("boxplot_pl"), type = 4)
      )
    )
  ),
  tabPanel(
    "Diagnostics",
    fluidRow(
      column(6, h4("Residuals vs Fitted"),
             withSpinner(plotlyOutput("resid_plot_pl"), type = 4)),
      column(6, h4("Normal Q-Q"),
             withSpinner(plotlyOutput("qq_plot_pl"), type = 4))
    ),
    br(),
    h4("Variance Inflation Factors"),
    DTOutput("vif_tab"),
    br(),
    h4("Assumption Tests"),
    DTOutput("assump_tab")
  ),
  tabPanel(
    "Cross-Validation",
    sidebarLayout(
      sidebarPanel(
        numericInput("kfold", "K folds", 5, min = 3, max = 10),
        actionBttn("run_cv", "Run CV", style = "jelly", color = "success", icon = icon("play"))
      ),
      mainPanel(
        div(class="app-gradient", HTML("<b>Tip:</b> Look for tight RMSE spread and consistently high R² across folds.")),
        DTOutput("cv_table")
      )
    )
  )
)

# ── Server ──────────────────────────────────────────────────────────────────
server <- function(input, output, session){
  
  fitted_vars  <- reactiveVal(NULL)
  fitted_input <- reactiveVal(list())
  
  output$dyn_inputs <- renderUI({
    req(input$vars)
    lapply(input$vars, function(v) mk_input(paste0("inp_", v), v, train_data))
  })
  
  observeEvent(input$fit, {
    req(input$vars)
    fitted_vars(input$vars)
    vals <- lapply(input$vars, function(v){
      val <- input[[paste0("inp_", v)]]
      if (is.numeric(train_data[[v]])) as.numeric(val) else as.character(val)
    })
    names(vals) <- input$vars
    fitted_input(vals)
    showNotification("Model fitted and prediction generated", type = "message", duration = 3)
  })
  
  fit_model <- eventReactive(input$fit, {
    vars <- req(fitted_vars())
    f <- as.formula(paste("Price ~", paste(vars, collapse = "+")))
    dat <- train_data
    for (v in vars) if (!is.numeric(dat[[v]])) dat[[v]] <- factor(dat[[v]])
    lm(f, data = dat)
  }, ignoreInit = TRUE)
  
  newdata_row <- reactive({
    vars <- fitted_vars(); req(vars)
    vals <- fitted_input()
    nd <- lapply(vars, function(v){
      x <- train_data[[v]]
      if (is.numeric(x)) {
        as.numeric(vals[[v]])
      } else {
        factor(as.character(vals[[v]]), levels = levels(factor(x)))
      }
    })
    names(nd) <- vars
    as.data.frame(nd, stringsAsFactors = FALSE)
  })
  
  output$pred_card <- renderUI({
    validate(need(input$fit > 0, "Click ‘Fit / Predict’ after selecting variables and entering values."))
    m  <- fit_model(); req(m)
    nd <- newdata_row(); req(nrow(nd) == 1)
    pr_conf <- predict(m, newdata = nd, interval = "confidence", level = input$conf)
    pr_pred <- predict(m, newdata = nd, interval = "prediction", level = input$conf)
    
    pred_txt <- sprintf("%s", fmt_dollar(pr_conf[1, 'fit']))
    ci_txt   <- sprintf("%s – %s", fmt_dollar(pr_conf[1,'lwr']), fmt_dollar(pr_conf[1,'upr']))
    pi_txt   <- sprintf("%s – %s", fmt_dollar(pr_pred[1,'lwr']), fmt_dollar(pr_pred[1,'upr']))
    
    tagList(
      div(class="pred-card",
          HTML(sprintf("
            <h3 style='margin-top:4px;'>Predicted Price: <span style='font-weight:800;'>%s</span></h3>
            <div style='margin-top:6px;'>
              <span class='badge badge-ci'>CI (%.0f%%): %s</span>
              &nbsp;&nbsp;
              <span class='badge badge-pi'>PI (%.0f%%): %s</span>
            </div>", pred_txt, input$conf*100, ci_txt, input$conf*100, pi_txt))
      )
    )
  })
  
  stat_tbl <- reactive({
    validate(need(input$fit > 0, "Fit a model to see statistics."))
    m <- fit_model(); req(m)
    y <- model.frame(m)$Price; yhat <- fitted(m)
    tibble::tibble(
      Metric = c("AIC","RMSE","MAE","R²"),
      Value  = c(AIC(m), sqrt(mean((y-yhat)^2)), mean(abs(y-yhat)), summary(m)$r.squared)
    )
  })
  
  output$kpi_cards <- renderUI({
    req(input$fit > 0)
    st <- stat_tbl()
    aic  <- round(st$Value[st$Metric=="AIC"], 1)
    rmse <- fmt_dollar(st$Value[st$Metric=="RMSE"])
    r2   <- round(st$Value[st$Metric=="R²"], 3)
    
    fluidRow(
      column(4, div(class="kpi kpi-aic",  h4("AIC"),  div(class="val", aic))),
      column(4, div(class="kpi kpi-rmse", h4("RMSE"), div(class="val", rmse))),
      column(4, div(class="kpi kpi-r2",  h4("R²"),   div(class="val", r2)))
    )
  })
  
  output$stat_table <- renderDT({
    datatable(
      stat_tbl() %>% mutate(Value = ifelse(Metric %in% c("RMSE","MAE"), fmt_dollar(Value), round(Value, 6))),
      rownames = FALSE,
      options = list(dom = "t", paging = FALSE)
    )
  })
  
  output$boxplot_pl <- renderPlotly({
    req(input$vars)
    v <- input$vars[1]
    p <- ggplot(train_data, aes_string(x = v, y = "Price")) +
      (if (is.numeric(train_data[[v]])) geom_point(alpha=0.6) else geom_boxplot(alpha=0.6)) +
      theme_minimal(base_family = "Inter") +
      labs(x = v, y = "Price", title = "Price relationship")
    
    ggplotly(p)
  })
  
  output$resid_plot_pl <- renderPlotly({
    validate(need(input$fit > 0, "Fit a model to view residual diagnostics."))
    m <- fit_model(); req(m)
    p <- ggplot(broom::augment(m), aes(.fitted, .resid)) +
      geom_point(alpha=0.55) + geom_hline(yintercept=0, lty=2) +
      theme_minimal(base_family = "Inter") + labs(x="Fitted", y="Residual", title="Residuals vs Fitted")
    ggplotly(p)
  })
  
  output$qq_plot_pl <- renderPlotly({
    validate(need(input$fit > 0, "Fit a model to view the Q-Q plot."))
    m <- fit_model(); req(m)
    qq <- qqnorm(resid(m), plot.it = FALSE)
    dfp <- data.frame(x=qq$x, y=qq$y)
    p <- ggplot(dfp, aes(x,y)) +
      geom_point(alpha=0.7) + geom_abline(slope=1, intercept=0, lty=2) +
      theme_minimal(base_family = "Inter") + labs(x="Theoretical", y="Sample", title="Normal Q-Q")
    ggplotly(p)
  })
  
  output$vif_tab <- renderDT({
    validate(need(input$fit > 0, "Fit a model to compute VIF."))
    m <- fit_model(); req(m)
    vif_vals <- tryCatch(car::vif(m), error=function(e) NA)
    if (all(is.na(vif_vals))) {
      datatable(data.frame(Message = "VIF not available for this model."), options = list(dom="t", paging=FALSE))
    } else {
      datatable(
        tibble::tibble(Variable = names(vif_vals), VIF = as.numeric(vif_vals)),
        rownames = FALSE, options = list(pageLength = 10, dom = "tip")
      )
    }
  })
  
  output$assump_tab <- renderDT({
    validate(need(input$fit > 0, "Fit a model to run assumption tests."))
    m <- fit_model(); req(m)
    bp <- lmtest::bptest(m); sw <- shapiro.test(resid(m))
    datatable(
      tibble::tibble(
        Test = c("Breusch–Pagan (Homosced.)","Shapiro–Wilk (Normality)"),
        Statistic = round(c(as.numeric(bp$statistic), as.numeric(sw$statistic)), 4),
        p_value   = signif(c(bp$p.value, sw$p.value), 4)
      ),
      rownames = FALSE, options = list(dom = "tip", pageLength = 5)
    )
  })
  
  output$cv_table <- renderDT({
    req(input$run_cv > 0)
    vars <- isolate(input$vars); validate(need(length(vars) > 0, "Select at least one variable."))
    f <- as.formula(paste("Price ~", paste(vars, collapse="+")))
    dat <- train_data
    for (v in vars) if (!is.numeric(dat[[v]])) dat[[v]] <- factor(dat[[v]])
    
    ctrl <- trainControl(method="cv", number=isolate(input$kfold), savePredictions = "final")
    set.seed(22)
    fit <- caret::train(f, data=dat, method="lm", trControl=ctrl)
    
    rmse_vals <- fit$resample$RMSE
    r2_vals   <- fit$resample$Rsquared
    
    summary_tbl <- tibble::tibble(
      Metric = c("RMSE (mean)", "RMSE (min)", "RMSE (max)", "R² (mean)", "R² (min)", "R² (max)"),
      Value  = c(mean(rmse_vals), min(rmse_vals), max(rmse_vals), mean(r2_vals), min(r2_vals), max(r2_vals))
    )
    
    fold_tbl <- tibble::tibble(
      Fold = paste0("Fold ", seq_along(rmse_vals)),
      `RMSE / R²` = paste0(round(rmse_vals,3), " / ", round(r2_vals,3))
    )
    
    datatable(
      bind_rows(
        summary_tbl %>% mutate(Value = ifelse(grepl("RMSE", Metric), fmt_dollar(Value), round(Value, 4))),
        tibble::tibble(Metric = fold_tbl$Fold, Value = fold_tbl$`RMSE / R²`)
      ),
      rownames = FALSE,
      options = list(dom = "tip", pageLength = 50)
    )
  })
}

shinyApp(ui, server)
