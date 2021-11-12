import React, { Component } from 'react';
import Comment from './Fragments/comment.js';
import Result from './Fragments/Result.js';
import ApiService from '../Utils/ApiService.js';

import './Test.css';

class Test extends Component {
    constructor(){
        super()
        this.state={
            comment: '',
            sentiment: '',
        }
    }
    onTextChange = (event) =>
    {
        this.setState({comment: event.target.value});
    }
    testComment_onClick = (event) =>
    {
        event.preventDefault();
        let COMMENT = {
            comment: this.state.comment
        }
        ApiService.postComment(COMMENT).then((res) =>{
            console.log(res);
            this.setState({sentiment: res.data})
        })
    }
    render() {
        const sentiment = this.state.sentiment;
        return (
            <div className="div_main">
                <Comment textChange={this.onTextChange} onTestClick={this.testComment_onClick}/>
                <Result comment={sentiment}/>
            </div>
        );
    }
}

export default Test;