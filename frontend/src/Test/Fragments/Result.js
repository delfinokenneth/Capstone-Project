import React from "react";
import '../Test.css';

const Result = (props) => 
{
    console.log(props.comment)
    return (  
        <div className="div_Result">
            <h2>Result</h2>
            <ul>
                {props.comment.map(comments => (
                    <li>
                        {
                            comments.map(c => (
                                <ul>{c}</ul>
                            ))
                        }
                    </li>
                ))}
            </ul>
        </div>

    );
}

export default Result;