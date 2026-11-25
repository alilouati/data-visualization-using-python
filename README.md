# Data Visualization using Python
*Ph.D. in Computing — hands-on in Google Colab*
Dr. Ali Louati · IS Dept., PSAU

Each chapter has its own folder with its slides (PDF) and its Colab notebook side by side. Every deck opens with a "This chapter's Colab lab" slide listing the notebook's sections, and every notebook section has matching slides.

```
Chapter_1/  Chapter_1_Introduction_PhD.pdf        Ch1_Introduction_Foundations.ipynb
Chapter_2/  Chapter_2_Plotting_Libraries_PhD.pdf  Ch2_Plotting_Libraries.ipynb
Chapter_3/  Chapter_3_3D_Plotting_PhD.pdf         Ch3_3D_and_Multivariate.ipynb
Chapter_4/  Chapter_4_Dashboard_PhD.pdf           Ch4_Dashboards_Dash_Bokeh.ipynb
Chapter_5/  Chapter_5_Streamlit_PhD.pdf           Ch5_Streamlit_Apps.ipynb   2022.csv
```

| Ch. | Slides (PDF) | Colab notebook | New PhD-level content |
|---|---|---|---|
| 1 | Chapter_1_Introduction_PhD.pdf | Ch1_Introduction_Foundations.ipynb | Anscombe, Cleveland–McGill & Stevens' law, bin/KDE sensitivity, over-plotting at 100k points, colour-map lightness (L*), colour-blindness simulation, lie factor, uncertainty (SD/SE/CI) |
| 2 | Chapter_2_Plotting_Libraries_PhD.pdf | Ch2_Plotting_Libraries.ipynb | OO Matplotlib, corrected slide code, logistic vs linear fit (Titanic), Simpson's paradox, Plotly animation, Altair brushing & linking, publication-ready vector figures |
| 3 | Chapter_3_3D_Plotting_PhD.pdf | Ch3_3D_and_Multivariate.ipynb | When 3-D helps/hurts, Himmelblau/Rastrigin, optimiser trajectories (GD/Momentum/Adam), real Maunga Whau data, isosurfaces, clustered heatmaps, PCA vs t-SNE |
| 4 | Chapter_4_Dashboard_PhD.pdf | Ch4_Dashboards_Dash_Bokeh.ipynb | Dash inline in Colab, KPI cards, chained callbacks, cross-filtering, `State`, unit-tested callbacks, serverless Bokeh (CustomJS), ipywidgets |
| 5 | Chapter_5_Streamlit_PhD.pdf | Ch5_Streamlit_Apps.ipynb | Streamlit from Colab (Cloudflare tunnel / Colab proxy), AppTest automated tests, caching & session state, GRAALI replica, Al-Kharj 2022 road-accident app with train/test split, leakage check, silhouette, deployment |

## Live dashboard (Chapter 5)
`Chapter_5/dashboard/streamlit_app.py` is the Al-Kharj road-accidents dashboard from the Chapter 5 notebook, ready to run online on Streamlit Community Cloud (main file path `Chapter_5/dashboard/streamlit_app.py`). Run it on your computer with `streamlit run Chapter_5/dashboard/streamlit_app.py`.

## Extra lab: Can we trust an AI's explanation?
`LLM_Explanation_Lab/` holds a short, beginner-friendly Colab lab: a toy traffic junction, an AI advisor that explains its choices, a check of whether those explanations are true, and a safety shield. [Open in Colab](https://colab.research.google.com/github/alilouati/data-visualization-using-python/blob/main/LLM_Explanation_Lab/LLM_Explanation_Lab.ipynb)

## How students open a notebook in Colab
**Option 1 — upload (works today):** go to https://colab.research.google.com → *File → Upload notebook* → choose the `.ipynb`. Or put the files in Google Drive and double-click → *Open with Google Colaboratory*.

**Option 2 — one-click badges:** push the `Chapter_N/` folders and this README to a public GitHub repo named `alilouati/data-visualization-using-python` (never the `instructor_only/` folder). The "Open in Colab" badge at the top of each notebook then works. Using a different repo name? Change `REPO` in the badge URLs.

Then: *Runtime → Run all*, work through the ✏️ exercises; 🔑 solution cells are collapsed (double-click to reveal); *File → Save a copy in Drive* to keep work.

## Data
`Chapter_5/2022.csv`: Al-Kharj daily road accidents, 2022 (365 days). The Chapter 5 notebook loads it automatically; in Colab you can also upload it when asked.
