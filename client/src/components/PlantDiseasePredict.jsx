import React, { useState } from "react";
import crop_left from "../assets/images/crop-left-dec.png";
import "./CropPredict.css";
import DiseaseCard from "./DiseaseCard";

const PlantDiseasePredict = () => {
  const [data, setData] = useState("");
  const [Loading, setLoading] = useState(-1);
  const [imageUrl, setImageUrl] = useState(null);

  const [file, setFile] = useState('');

  const handleFileChange = (e) => {
    if (e.target.files) {
      setFile(e.target.files[0]);
    }
  };

  const handleUploadClick = (e) => {

    e.preventDefault();
    const imageUrl = URL.createObjectURL(file);
    setImageUrl(imageUrl);
    setLoading(1);
    const formData = new FormData();

    formData.append('files', file);
    fetch('http://localhost:5000/predictdisease', {
      body: formData,
      method: 'POST',
      origin: 'http://localhost:3000'
    })
      .then(response => response.json())
      .then((data) => {
        console.log(data);
        setData(data);
        setLoading(0);
      })
      .catch(error => console.error("Error!", error));
  }

  console.log(data);

  return (
    <div id="disease" className="our-prediction section">
      <div className="prediction-left-dec">
        <img src={crop_left} alt="" />
      </div>
      <div className="container">
        <div className="row">
          <div className="col-lg-6 offset-lg-3">
            <div className="section-heading">
              <h2>
                Detect <em>Plant Diseases</em>
              </h2>
            </div>
          </div>
        </div>
        <form
          action="http://localhost:5000/predictdisease"
          className="crop-form"
          method="POST"
          encType="multipart/form-data"
        >
          <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <div className="inputDiv" style={{ width: '100%', justifyContent: 'center', alignItems: 'center' }}>
              <label htmlFor="nitrogen">Upload Leaf Image</label>
              <input onChange={handleFileChange} type="file" multiple required id="fileUpload" />
            </div>
            <div style={{ display: 'flex', justifyContent: 'center' }}>
              <button onClick={handleUploadClick} style={{ width: '20%' }} type="button">
                Submit
              </button>
            </div>
            {Loading === 1 ? (
              <div className="resultDiv">Loading....</div>
            ) : data !== "" ? (
              <div className="container-fluid">
                <div className="row">
                  <div className="col-lg-12">
                    {Object.entries(data).map(([key, value]) => (
                      <DiseaseCard data={value} img={imageUrl} item="0" key={2} />
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <span>No Results</span>
            )}
          </div>
        </form>
      </div>
    </div>
  );
};

export default PlantDiseasePredict;
