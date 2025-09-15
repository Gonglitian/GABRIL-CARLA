python3 /home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/draw_plot/CARLA_bar.py \
  --input /home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/draw_plot/data/Original.csv \
          /home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/draw_plot/data/Confounded.csv \
  --output-dir /home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/draw_plot/figures \
  --figsize 7.2 3.2 --dpi 600 \
  --title-size 10 --axis-label-size 6 --tick-size 6 --legend-size 5 --annot-size 5

python3 /home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/draw_plot/CARLA_curve.py \
--input /home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/draw_plot/data/table3.csv \
--output-dir /home/vla-reasoning/proj/vlm-gabril/GABRIL-CARLA/draw_plot/figures \
--figsize 3.5 2.6 --dpi 600 \
--title-size 10 --axis-label-size 6 --tick-size 6 --legend-size 4 --annot-size 5

python CARLA_classes_bar.py \
  --bar-width 0.3 \
  --bar-gap 0.005 \
  --left-margin 0.3 \
  --detection-color "#ADD8E6" \
  --vlm-color "#FFB6C1"\
  --top-k 15