import React from "react";
import '../Test.css';

const Result = (props) => 
{
    return (
        <div className="div_Result">
            <h2>Result</h2>
            <text>{props.comment}</text>
        </div>
    );
}

export default Result;