# Run this in your AdDetector folder
files = [
    "filler/cars_movie_230s.mp4",
    "ads_raw/amazon_20s.mp4",
    "ads_raw/hyundai_20s.mp4",
    "ads_raw/flipkart_20s.mp4",
    "ads_raw/renault_45s.mp4"
]

with open("broadcast_list.txt", "w") as f:
    for file in files:
        f.write(f"file '{file}'\n")

print("Done!")