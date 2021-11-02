import React from "react";
import '../Test.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import TestButton from './TestButton.js';

const inputComment = (props) =>
{
    return (
        <div className="div_Comment">
            <h2>Enter you comment here: </h2>
            <input 
            placeholder="Enter Comment" 
            className="input_Comment"
            onChange={props.textChange}
            />
            <TestButton onTestClick = {props.onTestClick}/>
        </div>
    );
}

export default inputComment;