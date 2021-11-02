import React from "react";
import {Button} from 'react-bootstrap';

const testButton = (props) =>
{
    return (
        <Button 
        variant="primary" 
        className="btn_test"
        onClick={props.onTestClick}
        >
            test
        </Button>
    );
}
export default testButton;