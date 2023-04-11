import React from "react";
import "./CropCard.css";

const DiseaseCard = (props) => {
    return (
        <div className="my-crops" style={{ width: '80%' }} data-aos="zoom-in-up" data-aos-duration="1500">
            <a href="#map">
                <h4 style={{ color: "#812000", fontWeight: '700', fontSize: '20px' }}>Predicted Disease</h4>
                <img style={{ width: '30%' }} src={props.img} alt="" />
                <h4 className="web">{props.data.prediction}</h4>
                <div style={{ background: '#fff', padding: '20px', borderRadius: '10px' }}>
                    <h5 style={{ textAlign: 'left', fontWeight: '600' }}>Brief Description</h5>
                    <p style={{ textAlign: 'left', fontWeight: '500', fontSize: '13px' }}>
                        {props.data.description}
                    </p>
                </div>
                <div style={{marginTop:'30px', display: 'flex', justifyContent: 'space-between' }}>
                    <div style={{ width: '52%', background: '#fff', padding: '18px', borderRadius: '10px' }}>
                        <h5 style={{ textAlign: 'left', fontWeight: '600' }}>Identification/Sysmptoms</h5>
                        <p style={{ textAlign: 'left', fontWeight: '500', fontSize: '13px' }}>
                            {props.data.symptoms}
                        </p>
                    </div>
                    <div style={{ width: '43%', background: '#fff', padding: '18px', borderRadius: '10px' }}>
                        <h5 style={{ textAlign: 'left', fontWeight: '600' }}>Disease Prevention/Treatment</h5>
                        <p style={{ textAlign: 'left', fontWeight: '500', fontSize: '13px' }}>
                            For more information, visit the <a href={props.data.source} target="_blank">source</a>
                        </p>
                    </div>
                </div>
            </a>
        </div>
    );
};

export default DiseaseCard;
