import React from "react";
import '../Test.css';

const Result = (props) => 
{
    console.log(props.comment)
    return (  
        <div className="div_Result">
            <h2>Result</h2>
            <ol>
                {props.comment.map(comments => (
                    <li>
                        {   
                          comments.map(line => (
                            <ul>
                                {line}
                            </ul>

                          ))
                        }
                    </li>
                ))}
            </ol>
        </div>

    );
}

export default Result;