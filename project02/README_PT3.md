# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Final Project, Part 3: Final notebook.

### PROMPT

Our goal for this project is to develop a working technical document that can be shared amongst your peers. Similar to any other technical project, it should surface your work and approach in a human readable format. Your project should push the reader to ask for more insightful questions, and avoid issues like, "what does this line of code do?" 

From a presentation perspective, think about the machine learning applications of your data. Use your model to display correlations, feature importance, and unexplained variance. Document your research with a summary, explaining your modeling approach as well as the strengths and weaknesses of any variables in the process. 

You should provide insight into your analysis, using best practices like cross validation or any applicable prediction metrics (ex: MSE for regression; Accuracy/AUC for classification). Remember, there are many metrics to choose from, so be sure to explain why the one you've used is reasonable for your problem. 

Look at how your model performs compared to a baseline model, and articulate the benefit gained by using your specific model to solve this problem. Finally, build visualizations that explain outliers and the relationships of your predicted parameter and independent variables. You might also identify areas where new data could help improve the model in the future.

**Goal:** Detailed iPython technical notebook with a summary of your statistical analysis, model, and evaluation metrics

---

### DELIVERABLES

#### iPython Report

- **Requirements:**
  - Create an iPython Notebook with code, visualizations, and markdown
  - Summarize your exploratory data analysis. 
  - Explain your choice of validation and prediction metrics.
  - Frame source code so it enhances your notebook explanations.
  - Include a separate python module with helper functions, if needed
    - Consider it like an appendix piece; although unlike an appendix, it'll be necessary for your project to function!
  - Visualize relationships between your Y and your two strongest variables, as determined by some scoring measure (p values and coefficients, gini/entropy, etc).
  - Identify areas where new data could help improve the model


- **Bonus:**
    - Many modeling approaches are all about fine-tuning the algorithm parameters and trying to find a specific value. Show how you optimized for this value, and the costs/benefits of doing so.

- **Tips:**
    - Aim to explain your modelling process with Markdown, in a way that someone could follow your work, just by reading the markdown and looking at the plots.
    
    - Try all the appropiate models for your problem (either classification or regression).
    
    - Create a good visual summary of your models performance.
    
    - It's very likely that you won't have enough time to try out everything you might want to. As part of your executive summary, list down what the appropiate next steps would be. 

---

### RESOURCES

#### Suggestions for Getting Started

- Two common ways to start models:
    -  "Kitchen Sink Strategy": throw all the variables in and subtract them out, one by one.
    -  "Single Variable Strategy": start with the most important variable and slowly add in while paying attention to performance)
        - It may be worth exploring both to understand your data and problem. How slow is building and predicting the model with all the variables? How much improvement is made with each variable added?
- Recall that your variables maybe need transformation in order to be most useful.
- Algorithms have different requirements (say, random forest vs logistic regression), and one may work better for your data than another.
- Strike a balance between writing, code, and visual aids. Your notebook should feel like a blogpost with some code in it. Force yourself to write and visualize more than you think!

#### Past Projects

- You can find previous General Assembly Presentations and Notebooks at the [GA Gallery](https://gallery.generalassemb.ly/DS?metro=).


### Assessment

This submission will be marked on a four point scale for each item and feedback will be provided for each:

0 - Doesn't meet expectations

1 - Partially meets expectations

2 - Meets expectations

3 - Exceeds expectations

The items to be assessed are:

* EDA summary plus conclusions - coding
* EDA summary plus conclusions - methodology
* EDA summary plus conclusions - conceptual understanding

* Full modelling - coding
* Full modelling - methodology
* Full modelling - conceptual understanding

As always, the key aspect in the assessment is to provide feedback for improvement and clarification.
