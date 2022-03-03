import React, { Component } from 'react';
import Comment from './fragments/commentbox.js';
import Result from './fragments/result.js';
import Questions from './fragments/questions.js';
import ApiService from '../Utils/ApiService.js';
import './evaluationForm.css';

class comment extends Component {
    constructor(){
        super()
        this.state ={
            comment: '',
            sentiment: '',
            questions_sec1: [],
            questions_sec2: [],
            questions_sec3: [],
            questions_sec4: [],
            questions_sec5: [],
        }
    }
    componentDidMount()
    {
        //get questions for section 1
        ApiService.getQuestions(1).then((res) => {
            this.setState({questions_sec1: res.data})
            console.log(this.state.questions_sec1)
        })
        //get questions for section 2
        ApiService.getQuestions(2).then((res) => {
            this.setState({questions_sec2: res.data})
            console.log(this.state.questions_sec2)
        })
        //get questions for section 3
        ApiService.getQuestions(3).then((res) => {
            this.setState({questions_sec2: res.data})
            console.log(this.state.questions_sec2)
        })
        //get questions for section 4
        ApiService.getQuestions(4).then((res) => {
            this.setState({questions_sec2: res.data})
            console.log(this.state.questions_sec2)
        })
        //get questions for section 5
        ApiService.getQuestions(5).then((res) => {
            this.setState({questions_sec2: res.data})
            console.log(this.state.questions_sec2)
        })

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
    componentDidMount
    render() {
        return ( 
            <div className="main_div">
                <Questions sec1={[this.state.questions_sec1]}></Questions>
                <Comment textChange={this.onTextChange} submitComment={this.onSubmitComment}/>
                <Result sentiment={this.state.sentiment}/>
            </div>
        );
    }
}

export default comment;