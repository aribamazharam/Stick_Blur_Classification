  <!DOCTYPE html>
<html>

<head>
</head>
<body>
<h1>Stick Blur Classification</h1>

  <h2>Table of Contents</h2>
  <ul>
    <li><a href="#stickIntroduction">Introduction</a></li>
    <li><a href="#stickDataset">Dataset</a></li>
    <li><a href="#stickResults">Results and Outcomes</a></li>
  </ul>

  <h2 id="Introduction">Introduction</h2>
  <p>Welcome to the Stick Blur or Not Classification project! The main goal of this project is to classify images into
    two categories: stick blur or not. To achieve this task, a deep learning model has been trained and evaluated.</p>

  <h2 id="Dataset">Dataset</h2>
  <p>The dataset used for this project was self-collected. It consists of images that have either stick blur or not.
    The dataset was prepared to facilitate the training and evaluation of the classification model.</p>

  <h2>Key Features:</h2>
  <ul>
    <li>Stick Blur or Not classification using EfficientNet model</li>
  </ul>

  <h2>Technologies Used:</h2>
  <ul>
    <li>Python for scripting and development</li>
    <li>PyTorch as the deep learning framework</li>
    <li>EfficientNet model for Stick Blur classification</li>
  </ul>
  
  <h2>RAM</h2>
  <ul>
    <li>The RAM utilization was 6GB</li>
  </ul>
  
  <h2 id="Results">Results and Outcomes:</h2>
  <p>The model was trained for 5 epochs and achieved the following results:</p>

  <table>
    <tr>
      <th>Epoch</th>
      <th>Train Loss</th>
      <th>Train Accuracy</th>
      <th>Validation Loss</th>
      <th>Validation Accuracy</th>
      <th>Test Accuracy</th>
      <th>Average Inference Time per Iteration</th>
    </tr>
    <tr>
      <td>5</td>
      <td>0.1277</td>
      <td>0.9729</td>
      <td>0.1004</td>
      <td>0.9667</td>
      <td>0.9355</td>
      <td>0.0473 seconds</td>
    </tr>
  </table>
</body>

</html>
