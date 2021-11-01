import React, { Component } from 'react';
import Comment from './Fragments/comment.js';
import Result from './Fragments/Result.js';

import './Test.css';

class Test extends Component {
    constructor(){
        super()
        this.state={
            comment: '',
        }
    }
    render() {
        return (
            <div className="div_main">
                <Comment />
                <Result />
            </div>
        );
    }
}

export default Test;