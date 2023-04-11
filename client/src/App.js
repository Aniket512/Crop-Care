import "./App.css";
import Map from "./components/Map";
import Intro from "./components/Intro";
import About from "./components/About";
import Header from "./components/Header";
import Footer from "./components/Footer";
import CropPredict from "./components/CropPredict";
import FertPredict from "./components/FertPredict";

import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import PlantDiseasePredict from "./components/PlantDiseasePredict";

function App() {
  return (
    <div className="App">
      <Router>
        <Header />
        <Routes>
          <Route
            path="/"
            element={
              <>
                <Intro />
                <About />
                <CropPredict />
                <PlantDiseasePredict/>
                <FertPredict />
                <Map />
              </>
            }
          />
        </Routes>
        <Footer />
      </Router>
    </div>
  );
}

export default App;
