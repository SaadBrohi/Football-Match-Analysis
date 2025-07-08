# ⚽ Football Match Analysis using YOLOv8, Deep Learning, and KMeans

A complete machine learning and computer vision project for analyzing football (soccer) match videos. This system detects **players**, **referees**, and the **ball**, tracks them frame-by-frame, clusters **players by t-shirt color** using KMeans, and visualizes **ball possession** and **camera movement** insights.

Built using **YOLOv8**, trained with **Roboflow**, and powered by **OpenCV**, and **Scikit-learn**.

---

## 📌 Features

- 🔍 **Object Detection**: Detect players, referees, and the ball using **YOLOv8**.
- 🏟️ **Field-Only Detection**: Custom YOLOv8 model trained using **Roboflow** to detect only **on-field** players and referees.
- 🔁 **Object Tracking**: Assigns unique IDs to players/referees/ball across frames.
- 🎽 **Team Classification**: Uses **KMeans clustering** on t-shirt pixels to assign players to **Team 1** or **Team 2**.
- 📊 **Ball Possession Stats**: Displays which team is controlling the ball (currently randomized, easily extendable).
- 🎥 **Camera Motion Estimation**: Simulates camera movement in X and Y direction.
- 📦 **Modular Code**: Structured for clarity and scalability.

---


---

## 🛠️ Tools & Technologies

| Tool         | Purpose |
|--------------|---------|
| YOLOv8       | Object detection (players, referees, ball) |
| Roboflow     | Dataset annotation, management, and export |
| OpenCV       | Video processing and visualization |
| Scikit-learn | KMeans clustering |
| PyTorch      | YOLOv8 training/inference backend |
| Git & GitHub | Version control and hosting |

---

## Output Summary

Each frame shows:

Bounding boxes with unique IDs for players, referees, and ball.

Players grouped by t-shirt color into Team 1 and Team 2.

Randomly generated ball possession percentages.

Simulated camera movement (Δx, Δy).
Each frame shows:

Bounding boxes with unique IDs for players, referees, and ball.

Players grouped by t-shirt color into Team 1 and Team 2.

Randomly generated ball possession percentages.

Simulated camera movement (Δx, Δy).

---

## 🔮 Future Improvements
Replace random ball possession with nearest-player-to-ball logic.

Integrate DeepSort for better tracking accuracy.

Add camera movement detection using optical flow.

Generate player heatmaps.

Streamlit dashboard for visualization.

## 🙋 Author
Saad Suleman Brohi
GitHub: @SaadBrohi
Email: saadbrohi008@gmail.com
Location: Karachi, Pakistan

## 📜 License
This project is licensed under the MIT License.
