import React from "react";
import '../Test.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import {Button} from 'react-bootstrap';

const inputComment = () =>
{
    return (
        <div className="div_Comment">
            <h2>Enter you comment here: </h2>
            <input placeholder="Enter Comment" className="input_Comment"></input>
            <Button variant="primary" className="btn_test">test</Button>
        </div>
    );
}

export default inputComment;