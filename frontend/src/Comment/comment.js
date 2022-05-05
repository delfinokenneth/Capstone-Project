import React, { Component } from 'react';
import Comment from './fragments/commentbox.js';
import Result from './fragments/result.js';
import ApiService from '../Utils/ApiService.js';

class comment extends Component {
    constructor(){
        super()
        this.state ={
            comment: '',
            sentiment: '',
        }
    }
    onTextChange = (event) =>
    {
        this.setState({comment: event.target.value});
        console.log(event.target.value);
    }
    onSubmitComment = (event) =>
    {
        event.preventDefault();
        let COMMENT = {
            comment: this.state.comment
        }
        console.log(COMMENT);
        ApiService.getSentiment(COMMENT).then((res) =>{
            console.log(res);
            this.setState({sentiment: res.data})
        })
    }
    render() {
        return (
            <div className="main_div">
                <Comment textChange={this.onTextChange} submitComment={this.onSubmitComment}/>
                <Result sentiment={this.state.sentiment}/>
            </div>
        );
    }
}

export default comment;